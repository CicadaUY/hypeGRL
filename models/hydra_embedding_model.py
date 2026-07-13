import pickle
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from models.base_hyperbolic_model import BaseHyperbolicModel
from models.hydra.hydra import hydra
from utils.geometric_conversions import convert_coordinates


class HydraModel(BaseHyperbolicModel):
    def __init__(self, config: Dict):
        self.dim = config.get("dim", 2)
        self.curvature = config.get("curvature", 1)
        self.alpha = config.get("alpha", 1.1)
        self.equi_adj = config.get("equi_adj", 0.5)

    @property
    def native_space(self) -> str:
        """Get the native embedding space for this model."""
        return "spherical"

    def train(
        self,
        edge_list: Optional[List[Tuple[str, str]]] = None,
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

        # Build distance matrix
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        n = G.number_of_nodes()
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i in lengths and j in lengths[i]:
                    D[i, j] = lengths[i][j]

        # Run Hydra
        self.embeddings = hydra(D, self.dim, self.curvature, self.alpha, self.equi_adj)

        # Save embeddings
        with open(model_path, "wb") as f:
            pickle.dump(self.embeddings, f)

    def get_all_embeddings(self, model_path: Optional[str] = None) -> np.ndarray:
        if model_path:
            with open(model_path, "rb") as f:
                self.embeddings = pickle.load(f)

        theta = self.embeddings["theta"]
        radius = self.embeddings["r"]

        r = np.arctanh(radius) * 2
        # Return spherical coordinates as [radius, theta]
        _embeddings = np.column_stack([r, theta])

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

        theta = self.embeddings["theta"]
        radius = self.embeddings["r"]
        r = np.arctanh(radius) * 2

        spherical_coords = np.column_stack([r, theta])
        return convert_coordinates(spherical_coords, "spherical", "hyperboloid")

    def to_poincare(self, model_path: Optional[str] = None) -> np.ndarray:
        """Convert spherical embeddings to Poincaré coordinates."""
        if model_path:
            with open(model_path, "rb") as f:
                self.embeddings = pickle.load(f)

        theta = self.embeddings["theta"]
        radius = self.embeddings["r"]
        r = np.arctanh(radius) * 2

        spherical_coords = np.column_stack([r, theta])
        return convert_coordinates(spherical_coords, "spherical", "poincare")
