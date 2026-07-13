import logging
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base_hyperbolic_model import BaseHyperbolicModel
from models.lorentz.lorentz import RSGD, Graph, Lorentz, recon
from utils.geometric_conversions import hyperboloid_to_poincare

# Enable logging for gensim
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)


class LorentzEmbeddingsModel(BaseHyperbolicModel):
    def __init__(self, config: dict):
        self.dim = config.get("dim", 2)
        self.epochs = config.get("epochs", 1000)
        self.batch_size = config.get("batch_size", 1)
        self.sample_size = config.get("sample_size", 5)
        self.learning_rate = config.get("learning_rate", 0.5)
        self.burn_epochs = config.get("burn_epochs", 10)
        self.burn_c = config.get("burn_c", 10)
        self.loader_workers = config.get("loader_workers", 2)
        self.logdir = config.get("logdir", "runs")
        self.save_step = config.get("save_step", 100)
        self.shuffle = config.get("shuffle", True)
        self.model = None
        self.num_nodes = config.get("num_nodes", None)
        self.node_lookup = None  # Will be set during training

    @property
    def native_space(self) -> str:
        """Get the native embedding space for this model."""
        return "hyperboloid"

    def train(
        self,
        edge_list: Optional[List[tuple]] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        model_path: str = "saved_models/model.bin",
    ):

        if edge_list is None and adjacency_matrix is None:
            raise ValueError("You must provide either edge_list or adjacency_matrix.")

        if edge_list is not None:
            # Extract node set and map to integer IDs
            node_ids = sorted(set(str(u) for u, v in edge_list) | set(str(v) for u, v in edge_list))
            id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
            self.node_lookup = {idx: node_id for node_id, idx in id_to_index.items()}  # For decoding later

            size = len(node_ids)
            adjacency_matrix = np.zeros((size, size), dtype=int)
            for u, v in edge_list:
                adjacency_matrix[id_to_index[str(u)], id_to_index[str(v)]] = 1

        self.num_nodes = adjacency_matrix.shape[0]
        self.model = Lorentz(n_items=self.num_nodes, dim=self.dim + 1)  # Lorentz space is (n+1)-dimensional
        dataset = Graph(adjacency_matrix, sample_size=self.sample_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.loader_workers)

        self.writer = SummaryWriter(f"{self.logdir}/{datetime.utcnow()}")
        self.optimizer = RSGD(self.model.parameters(), learning_rate=self.learning_rate)

        with tqdm(total=self.epochs, ncols=80, desc="Training", dynamic_ncols=True) as pbar:
            for epoch in range(self.epochs):
                if hasattr(self, "optimizer") and self.optimizer is not None:
                    self.optimizer.learning_rate = self.learning_rate / self.burn_c if epoch < self.burn_epochs else self.learning_rate
                for inputs, Ks in dataloader:
                    self.optimizer.zero_grad()
                    loss = self.model(inputs, Ks).mean()
                    loss.backward()
                    self.optimizer.step()

                self.writer.add_scalar("loss", loss, epoch)
                self.writer.add_scalar("recon_perf", recon(self.model.get_lorentz_table(), adjacency_matrix), epoch)
                self.writer.add_scalar("table_test", self.model._test_table(), epoch)

                if epoch % self.save_step == 0:
                    tmp_path = model_path + ".tmp"
                    torch.save(self.model.state_dict(), tmp_path)
                    os.replace(tmp_path, model_path)

                # Update progress bar
                pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}" if "loss" in locals() else f"Epoch {epoch+1}")
                pbar.update(1)

        logger.info("Training completed.")

        tmp_path = model_path + ".tmp"
        torch.save(self.model.state_dict(), tmp_path)
        os.replace(tmp_path, model_path)
        logger.info(f"Model saved in {model_path}.")

    def get_embedding(self, node_id: str, model_path: Optional[str] = None) -> np.ndarray:
        if model_path:
            self.model = Lorentz(n_items=self.num_nodes, dim=self.dim + 1)
            self.model.load_state_dict(torch.load(model_path))

        # Handle node ID lookup if we have a lookup table
        if self.node_lookup is not None:
            # Find the index for this node_id
            node_index = None
            for idx, stored_node_id in self.node_lookup.items():
                if stored_node_id == node_id:
                    node_index = idx
                    break
            if node_index is None:
                raise ValueError(f"Node ID '{node_id}' not found in the trained model")
        else:
            # Fallback to assuming node_id is already an integer index
            node_index = int(node_id)

        # +1 due to padding idx (index 0 is reserved for padding)
        lorentz_embedding = self.model.get_lorentz_table()[node_index + 1]
        # Return embedding in reordered format (standard hyperboloid format)
        return self._reorder_lorentz_coordinates(lorentz_embedding.reshape(1, -1))[0]

    def get_all_embeddings(self, model_path: Optional[str] = None) -> np.ndarray:
        if model_path:
            self.model = Lorentz(n_items=self.num_nodes, dim=self.dim + 1)
            self.model.load_state_dict(torch.load(model_path))

        # Get all embeddings except the padding (index 0)
        lorentz_table = self.model.get_lorentz_table()
        lorentz_embeddings = lorentz_table[1:]  # Skip padding index

        # Return embeddings in reordered format (standard hyperboloid format)
        return self._reorder_lorentz_coordinates(lorentz_embeddings)

    def _reorder_lorentz_coordinates(self, lorentz_coords: np.ndarray) -> np.ndarray:
        """
        Reorder Lorentz coordinates from (time, spatial) to (spatial, time) format.

        The Lorentz model uses the first coordinate as time, but the standard hyperboloid
        model uses the last coordinate as time. This function reorders the coordinates.

        Parameters:
        - lorentz_coords: Coordinates in Lorentz format (time first)

        Returns:
        - hyperboloid_coords: Coordinates in standard hyperboloid format (time last)
        """
        if lorentz_coords.shape[1] < 2:
            return lorentz_coords

        # Reorder: [time, x1, x2, ...] -> [x1, x2, ..., time]
        time_coord = lorentz_coords[:, 0:1]  # Keep as column vector
        spatial_coords = lorentz_coords[:, 1:]
        return np.concatenate([spatial_coords, time_coord], axis=1)

    def most_similar(self, node_id: str, topn: int = 5, model_path: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Find the most similar nodes to a given node using Lorentz distance.

        Parameters:
        - node_id: String identifier of the query node
        - topn: Number of most similar nodes to return
        - model_path: Optional path to load the model from

        Returns:
        - List of (node_id, similarity_score) tuples, ordered by similarity (highest first)
        """
        if model_path:
            self.model = Lorentz(n_items=self.num_nodes, dim=self.dim + 1)
            self.model.load_state_dict(torch.load(model_path))

        # Get the query node embedding (already in reordered format)
        query_embedding = self.get_embedding(node_id, model_path)

        # Get all embeddings (already in reordered format)
        all_embeddings = self.get_all_embeddings(model_path)

        # Calculate Lorentz distances (negative Lorentz scalar product)
        similarities = []
        for i, embedding in enumerate(all_embeddings):
            # Lorentz scalar product: <x,y> = x₁y₁ + ... + xₙyₙ - x₀y₀ (time coordinate last)
            # Distance is -<x,y> (negative because we want similarity, not distance)
            spatial_product = np.sum(query_embedding[:-1] * embedding[:-1])
            time_product = query_embedding[-1] * embedding[-1]
            lorentz_product = spatial_product - time_product
            similarity = -lorentz_product  # Negative for similarity

            # Get the actual node ID
            if self.node_lookup is not None:
                actual_node_id = self.node_lookup.get(i, str(i))
            else:
                actual_node_id = str(i)

            similarities.append((actual_node_id, float(similarity)))

        # Sort by similarity (highest first) and return topn
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

    def to_hyperboloid(self, model_path: Optional[str] = None) -> np.ndarray:
        """Return embeddings in hyperboloid coordinates (already in standard format)."""
        return self.get_all_embeddings(model_path)

    def to_poincare(self, model_path: Optional[str] = None) -> np.ndarray:
        """Convert hyperboloid embeddings to Poincaré coordinates."""
        hyperboloid_embeddings = self.get_all_embeddings(model_path)
        return hyperboloid_to_poincare(hyperboloid_embeddings)
