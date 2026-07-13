import logging
import pickle
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from models.base_hyperbolic_model import BaseHyperbolicModel
from models.hypermap.python import pyhypermap
from utils.geometric_conversions import convert_coordinates

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class HypermapEmbeddingModel(BaseHyperbolicModel):
    def __init__(self, config: Dict):
        self.k_min = config.get("k_min", 3)
        self.T = config.get("T", 0.5)
        self.zeta = config.get("zeta", 1.0)
        self.k_speedup = config.get("k_speedup", 0)
        self.m_in = config.get("m_in", -1)
        self.L_in = config.get("L_in", -1)
        self.corrections = config.get("corrections", True)
        self._edge_file = "tmp/data.edges"
        self.logger = logging.getLogger(__name__)

    @property
    def native_space(self) -> str:
        """Get the native embedding space for this model."""
        return "spherical"

    def train(
        self,
        edge_list: Optional[List[tuple]] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        model_path: str = "saved_models/model.bin",
    ):

        if edge_list is not None:
            n = max(max(u, v) for u, v in edge_list) + 1
            G = nx.DiGraph()
            G.add_edges_from(edge_list)
            G.add_nodes_from(range(n))
        elif adjacency_matrix is not None:
            G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
        else:
            raise ValueError("You must provide either edge_list or adjacency_matrix.")

        gamma_hat, _ = pyhypermap.estimate_gamma(G, k_min=self.k_min)
        coords_raw = pyhypermap.embed_from_nxgraph(
            G, gamma_hat, T=self.T, zeta=self.zeta, k_speedup=self.k_speedup, m_in=self.m_in, L_in=self.L_in, corrections=self.corrections
        )
        self.embeddings = np.array([(tple[1], tple[2]) for tple in coords_raw])

        # Save embeddings
        with open(model_path, "wb") as f:
            pickle.dump(self.embeddings, f)

    def get_all_embeddings(self, model_path: Optional[str] = None) -> np.ndarray:
        if model_path:
            with open(model_path, "rb") as f:
                self.embeddings = pickle.load(f)

        radius = self.embeddings[:, 1]
        theta = self.embeddings[:, 0]

        # Return spherical coordinates as [radius, theta]
        _embeddings = np.column_stack([radius, theta])

        return _embeddings

    def get_embedding(self, node_id: str, model_path: Optional[str] = None) -> np.ndarray:
        embeddings = self.get_all_embeddings(model_path)
        return embeddings[int(node_id)]

    def most_similar(self, node_id: str, topn: int = 5, model_path: Optional[str] = None) -> List[Tuple[str, float]]:
        pass

    def to_hyperboloid(self, model_path: Optional[str] = None) -> np.ndarray:
        """Convert spherical embeddings to hyperboloid coordinates."""
        if model_path:
            with open(model_path, "rb") as f:
                self.embeddings = pickle.load(f)

        radius = self.embeddings[:, 1]
        theta = self.embeddings[:, 0]
        spherical_coords = np.column_stack([radius, theta])
        return convert_coordinates(spherical_coords, "spherical", "hyperboloid")

    def to_poincare(self, model_path: Optional[str] = None) -> np.ndarray:
        """Convert spherical embeddings to Poincaré coordinates."""
        if model_path:
            with open(model_path, "rb") as f:
                self.embeddings = pickle.load(f)

        radius = self.embeddings[:, 1]
        theta = self.embeddings[:, 0]
        spherical_coords = np.column_stack([radius, theta])
        return convert_coordinates(spherical_coords, "spherical", "poincare")
