from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dmercator
import numpy as np

from models.base_hyperbolic_model import BaseHyperbolicModel
from utils.geometric_conversions import convert_coordinates


class DMercatorModel(BaseHyperbolicModel):
    def __init__(self, config: Dict):
        self.dim = config.get("dim", 1)
        self.beta = config.get("beta", -1)
        self._edge_file = "tmp/data.edges"

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
        Path(self._edge_file).parent.mkdir(parents=True, exist_ok=True)
        if edge_list is not None:
            with open(self._edge_file, "w") as f:
                for u, v in edge_list:
                    f.write(f"{u} {v}\n")
        elif adjacency_matrix is not None:
            rows, cols = np.where(adjacency_matrix)
            with open(self._edge_file, "w") as f:
                for u, v in zip(rows, cols):
                    f.write(f"{u} {v}\n")
        else:
            raise ValueError("You must provide either edge_list or adjacency_matrix.")

        # Run d-mercator
        dmercator.embed(edgelist_filename=self._edge_file, output_name=model_path, dimension=self.dim, beta=self.beta)

        self.embeddings_path = model_path + ".inf_coord"

    def get_all_embeddings(self, model_path: Optional[str] = None) -> np.ndarray:
        """
        Get all embeddings in spherical coordinates (radius, theta).

        Parameters:
        - model_path: Optional path to load the model from (if not already loaded)

        Returns:
        - np.ndarray: Array of shape (n_nodes, 2) containing [radius, theta] for each node
        """
        if model_path:
            self.embeddings_path = model_path + ".inf_coord"

        theta = np.loadtxt(self.embeddings_path, usecols=[2])
        radius = np.loadtxt(self.embeddings_path, usecols=[3])

        # Return spherical coordinates as [radius, theta]
        embeddings = np.column_stack([radius, theta])

        return embeddings

    def get_embedding(self, node_id: str, model_path: Optional[str] = None) -> np.ndarray:
        embeddings = self.get_all_embeddings(model_path)
        return embeddings[int(node_id)]

    def most_similar(self, node_id: str, topn: int = 5, model_path: Optional[str] = None) -> List[Tuple[str, float]]:
        pass

    def to_hyperboloid(self, model_path: Optional[str] = None) -> np.ndarray:
        """Convert spherical embeddings to hyperboloid coordinates."""
        spherical_embeddings = self.get_all_embeddings(model_path)
        # spherical_embeddings is [radius, theta], need to convert to hyperboloid
        return convert_coordinates(spherical_embeddings, "spherical", "hyperboloid")

    def to_poincare(self, model_path: Optional[str] = None) -> np.ndarray:
        """Convert spherical embeddings to Poincaré coordinates."""
        spherical_embeddings = self.get_all_embeddings(model_path)
        return convert_coordinates(spherical_embeddings, "spherical", "poincare")
