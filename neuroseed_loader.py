"""
NeuroSEED Dataset Loader

This module provides a loader for the NeuroSEED dataset, which contains
pre-computed hyperbolic embeddings and labels for various biological datasets.

The NeuroSEED dataset is from the paper:
"NeuroSEED: Geometric Deep Learning for Sequence-to-Sequence Tasks"
https://github.com/gcorso/NeuroSEED

Based on: https://github.com/pchlenski/hyperdt/blob/main/hyperdt/legacy/dataloaders/neuroseed.py
"""

import os
from typing import Tuple

import anndata
import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class NeuroSEEDLoader:
    """
    Loader for NeuroSEED dataset from h5ad files.

    The NeuroSEED dataset provides pre-computed hyperbolic embeddings
    for biological sequence data (e.g., American Gut microbiome data).
    """

    def __init__(self, data_file: str = "data/neuroseed/americangut.h5ad"):
        """
        Initialize the NeuroSEED loader.

        Parameters:
        -----------
        data_file : str
            Path to the h5ad file containing the NeuroSEED dataset
        """
        self.data_file = data_file
        self._adata = None

    def _load_adata(self):
        """Load the h5ad file if not already loaded."""
        if self._adata is None:
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(
                    f"Could not find NeuroSEED data file at {self.data_file}. "
                    f"Please ensure the NeuroSEED dataset is properly downloaded."
                )
            print(f"Loading NeuroSEED data from {self.data_file}...")
            self._adata = anndata.read_h5ad(self.data_file)
            print(f"Loaded data with shape: {self._adata.shape}")

    def get_data(
        self,
        seed: int,
        dimension: int,
        num_samples: int = 1250,
        convert_to_poincare: bool = True,
        min_label_count: int = 1000,
        taxonomy_level: str = "taxonomy_1",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train/test split of NeuroSEED data.

        Parameters:
        -----------
        seed : int
            Random seed for reproducibility
        dimension : int
            Dimensionality of the hyperbolic embeddings
        num_samples : int
            Number of samples to draw
        convert_to_poincare : bool
            If True, use Poincaré ball coordinates; if False, use hyperboloid
        min_label_count : int
            Minimum number of samples per label to keep (for filtering)
        taxonomy_level : str
            Taxonomy level to use for labels (e.g., "taxonomy_1")

        Returns:
        --------
        X_train : np.ndarray
            Training embeddings
        X_test : np.ndarray
            Testing embeddings
        y_train : np.ndarray
            Training labels
        y_test : np.ndarray
            Testing labels
        """
        self._load_adata()

        # Keep only abundant taxa
        labels = self._adata.var[taxonomy_level]
        labels_counts = labels.value_counts()
        keep = labels_counts[labels_counts > min_label_count].index
        labels_filtered = labels[labels.isin(keep)]

        print(f"Filtered labels: {len(labels_filtered)} samples with {len(keep)} classes")
        print(f"Classes: {keep.tolist()}")

        # Set seed and get indices randomly
        np.random.seed(seed)

        # Draw indices from filtered labels
        indices = np.random.choice(labels_filtered.index, num_samples, replace=False)

        # Get embeddings
        if convert_to_poincare:
            embed_name = f"component_embeddings_poincare_{dimension}"
        else:
            embed_name = f"component_embeddings_hyperboloid_{dimension}"

        if embed_name not in self._adata.varm:
            raise ValueError(f"Embedding '{embed_name}' not found in dataset. " f"Available embeddings: {list(self._adata.varm.keys())}")

        data = self._adata.varm[embed_name].loc[indices]
        labels = self._adata.var[taxonomy_level].astype("category").cat.codes.loc[indices]

        # Convert to numpy if needed
        if hasattr(data, "values"):
            data = data.values
        if hasattr(labels, "values"):
            labels = labels.values

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=seed)

        return X_train, X_test, y_train, y_test

    def get_training_data(
        self, class_label: int, seed: int, num_samples: int = 1250, convert_to_poincare: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get training data from NeuroSEED dataset.

        Parameters:
        -----------
        class_label : int
            Dimensionality of the hyperbolic space (serves as the "class label")
        seed : int
            Random seed
        num_samples : int
            Total number of samples (before train/test split)
        convert_to_poincare : bool
            If True, returns data in Poincaré ball coordinates
            If False, returns data in hyperboloid coordinates

        Returns:
        --------
        data : torch.Tensor
            Training embeddings
        labels : torch.Tensor
            Training labels
        """
        X_train, _, y_train, _ = self.get_data(
            seed=seed, dimension=class_label, num_samples=num_samples, convert_to_poincare=convert_to_poincare
        )
        return torch.as_tensor(X_train), torch.as_tensor(y_train, dtype=int).flatten()

    def get_testing_data(
        self, class_label: int, seed: int, num_samples: int = 1250, convert_to_poincare: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get testing data from NeuroSEED dataset.

        Parameters:
        -----------
        class_label : int
            Dimensionality of the hyperbolic space (serves as the "class label")
        seed : int
            Random seed
        num_samples : int
            Total number of samples (before train/test split)
        convert_to_poincare : bool
            If True, returns data in Poincaré ball coordinates
            If False, returns data in hyperboloid coordinates

        Returns:
        --------
        data : torch.Tensor
            Testing embeddings
        labels : torch.Tensor
            Testing labels
        """
        _, X_test, _, y_test = self.get_data(
            seed=seed, dimension=class_label, num_samples=num_samples, convert_to_poincare=convert_to_poincare
        )
        return torch.as_tensor(X_test), torch.as_tensor(y_test, dtype=int).flatten()

    def embeddings_to_graph(
        self, embeddings: np.ndarray, labels: np.ndarray, k_neighbors: int = 10, distance_metric: str = "poincare"
    ) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
        """
        Convert embeddings to a k-NN graph.

        Parameters:
        -----------
        embeddings : np.ndarray
            Node embeddings (N, D)
        labels : np.ndarray
            Node labels (N,)
        k_neighbors : int
            Number of nearest neighbors for graph construction
        distance_metric : str
            Distance metric to use ("poincare" or "euclidean")

        Returns:
        --------
        graph : nx.Graph
            NetworkX graph constructed from k-NN
        labels : np.ndarray
            Node labels
        node_indices : np.ndarray
            Node indices
        """
        print(f"Converting embeddings to k-NN graph (k={k_neighbors})...")

        num_nodes = len(embeddings)
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))

        # Compute pairwise distances
        if distance_metric == "poincare":
            distances = self._compute_poincare_distances(embeddings)
        else:
            from sklearn.metrics.pairwise import euclidean_distances

            distances = euclidean_distances(embeddings)

        # For each node, connect to k nearest neighbors
        for i in range(num_nodes):
            # Get k+1 nearest neighbors (excluding self)
            nearest_indices = np.argsort(distances[i])[1 : k_neighbors + 1]
            for j in nearest_indices:
                graph.add_edge(i, j)

        node_indices = np.arange(num_nodes)

        print(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        # Check connectivity
        if not nx.is_connected(graph):
            num_components = nx.number_connected_components(graph)
            largest_cc = max(nx.connected_components(graph), key=len)
            print(f"Warning: Graph has {num_components} connected components")
            print(f"Largest component has {len(largest_cc)} nodes ({100*len(largest_cc)/num_nodes:.1f}%)")

        return graph, labels, node_indices

    def _compute_poincare_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Poincaré distances between embeddings.

        Parameters:
        -----------
        embeddings : np.ndarray
            Embeddings in Poincaré disk (N, D)

        Returns:
        --------
        distances : np.ndarray
            Pairwise distance matrix (N, N)
        """
        # Import here to avoid circular dependency
        from utils.geometric_conversions import HyperbolicConversions

        distances = HyperbolicConversions.compute_distances(embeddings, space="poincare")
        return distances


# ============================================================================
# Standalone helper functions (compatible with hororf_benchmarks.py)
# ============================================================================


def get_training_data(class_label: int, seed: int, num_samples: int = 1250, convert_to_poincare: bool = True):
    """
    Get training data from NeuroSEED dataset.

    Parameters:
    -----------
    class_label : int
        Dimensionality of the hyperbolic space
    seed : int
        Random seed
    num_samples : int
        Total number of samples (before train/test split)
    convert_to_poincare : bool
        If True, returns data in Poincaré ball coordinates

    Returns:
    --------
    data : torch.Tensor
        Training embeddings
    labels : torch.Tensor
        Training labels
    """
    loader = NeuroSEEDLoader()
    return loader.get_training_data(class_label, seed, num_samples, convert_to_poincare)


def get_testing_data(class_label: int, seed: int, num_samples: int = 1250, convert_to_poincare: bool = True):
    """
    Get testing data from NeuroSEED dataset.

    Parameters:
    -----------
    class_label : int
        Dimensionality of the hyperbolic space
    seed : int
        Random seed
    num_samples : int
        Total number of samples (before train/test split)
    convert_to_poincare : bool
        If True, returns data in Poincaré ball coordinates

    Returns:
    --------
    data : torch.Tensor
        Testing embeddings
    labels : torch.Tensor
        Testing labels
    """
    loader = NeuroSEEDLoader()
    return loader.get_testing_data(class_label, seed, num_samples, convert_to_poincare)


def get_space():
    """Return the space type."""
    return "hyperbolic"
