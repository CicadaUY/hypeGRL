from abc import ABC, abstractmethod, abstractproperty
from typing import List, Optional, Tuple

import numpy as np


class BaseHyperbolicModel(ABC):
    """
    Abstract base class for hyperbolic embedding models.

    This class defines the interface that all hyperbolic embedding models must implement.
    It provides abstract methods for training, embedding retrieval, similarity computation,
    and coordinate system conversions.
    """

    @abstractproperty
    def native_space(self) -> str:
        """
        Get the native embedding space for this model.

        Returns:
        - str: The native coordinate space (e.g., "poincare", "hyperboloid", "spherical")
        """
        pass

    @abstractmethod
    def train(
        self,
        edge_list: Optional[List[tuple]] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        model_path: str = "saved_models/model.bin",
    ):
        """
        Train the hyperbolic embedding model.

        Parameters:
        - edge_list: Optional list of edges as tuples (source, target)
        - adjacency_matrix: Optional adjacency matrix representation of the graph
        - features: Optional node features array
        - model_path: Path where the trained model will be saved

        Raises:
        - ValueError: If neither edge_list nor adjacency_matrix is provided
        """
        pass

    @abstractmethod
    def get_embedding(self, node_id: str, model_path: Optional[str] = None) -> np.ndarray:
        """
        Get the embedding vector for a specific node.

        Parameters:
        - node_id: String identifier of the node
        - model_path: Optional path to load the model from (if not already loaded)

        Returns:
        - np.ndarray: Embedding vector for the specified node

        Raises:
        - KeyError: If the node_id does not exist in the model
        """
        pass

    @abstractmethod
    def get_all_embeddings(self, model_path: Optional[str] = None) -> np.ndarray:
        """
        Get embeddings for all nodes in the model.

        Parameters:
        - model_path: Optional path to load the model from (if not already loaded)

        Returns:
        - np.ndarray: Array of shape (n_nodes, embedding_dim) containing all embeddings
        """
        pass

    @abstractmethod
    def most_similar(self, node_id: str, topn: int = 5, model_path: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Find the most similar nodes to a given node.

        Parameters:
        - node_id: String identifier of the query node
        - topn: Number of most similar nodes to return
        - model_path: Optional path to load the model from (if not already loaded)

        Returns:
        - List[Tuple[str, float]]: List of (node_id, similarity_score) tuples,
          ordered by similarity (highest first)

        Raises:
        - KeyError: If the node_id does not exist in the model
        """
        pass

    @abstractmethod
    def to_hyperboloid(self, model_path: Optional[str] = None) -> np.ndarray:
        """
        Convert embeddings to hyperboloid coordinates.

        The hyperboloid model represents hyperbolic space as a hyperboloid
        in Minkowski space, where points satisfy x₀² - x₁² - ... - xₙ² = 1.

        Parameters:
        - model_path: Optional path to load the model from (if not already loaded)

        Returns:
        - np.ndarray: Embeddings in hyperboloid coordinates with shape (n_nodes, embedding_dim+1)
        """
        pass

    @abstractmethod
    def to_poincare(self, model_path: Optional[str] = None) -> np.ndarray:
        """
        Convert embeddings to Poincaré coordinates.

        The Poincaré model represents hyperbolic space as the interior of a unit ball,
        where distances are measured using the Poincaré metric.

        Parameters:
        - model_path: Optional path to load the model from (if not already loaded)

        Returns:
        - np.ndarray: Embeddings in Poincaré coordinates with shape (n_nodes, embedding_dim)
        """
        pass
