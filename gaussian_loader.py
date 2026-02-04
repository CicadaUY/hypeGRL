"""
Gaussian Hyperbolic Dataset Loader

This module provides a loader for synthetic Gaussian data in hyperbolic space,
using wrapped normal distributions. The data is generated using the hyperdt
library's wrapped_normal_mixture function.

Based on: https://github.com/pchlenski/hyperdt/blob/main/hyperdt/legacy/dataloaders/gaussian.py
"""

import os
from typing import Tuple

import networkx as nx
import numpy as np


class GaussianLoader:
    """
    Loader for Gaussian hyperbolic embeddings.

    Generates synthetic data using wrapped normal distributions in hyperbolic space.
    For the purposes of KNN classification, we construct a graph from these embeddings.
    """

    def __init__(self, data_dir: str = "./data/gaussian"):
        """
        Initialize the Gaussian loader.

        Parameters:
        -----------
        data_dir : str
            Directory to cache/save generated Gaussian dataset files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def _wrapped_normal_mixture(
        self,
        num_dims: int,
        num_classes: int,
        num_points: int,
        seed: int,
        noise_std: float = 2.0,
        separation: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a mixture of wrapped normal distributions in hyperbolic space.

        This is a simplified implementation based on hyperdt's wrapped_normal_mixture.

        Parameters:
        -----------
        num_dims : int
            Dimensionality of the hyperbolic space (Poincaré ball)
        num_classes : int
            Number of classes/clusters
        num_points : int
            Total number of points to generate
        seed : int
            Random seed
        noise_std : float
            Standard deviation of noise for cluster spread
        separation : float
            Separation between cluster centers

        Returns:
        --------
        data : np.ndarray
            Points in hyperboloid coordinates (N, D+1)
        labels : np.ndarray
            Class labels (N,)
        """
        np.random.seed(seed)

        points_per_class = num_points // num_classes

        # Generate cluster centers in tangent space at origin
        # Distribute them evenly in a sphere
        angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
        centers = []

        if num_dims == 2:
            # 2D case
            for angle in angles:
                center = separation * np.array([np.cos(angle), np.sin(angle)])
                centers.append(center)
        else:
            # Higher dimensions - distribute on unit sphere scaled by separation
            for i in range(num_classes):
                center = np.random.randn(num_dims)
                center = center / np.linalg.norm(center) * separation
                centers.append(center)

        all_points = []
        all_labels = []

        for class_idx, center in enumerate(centers):
            # Generate points around each center in tangent space
            for _ in range(points_per_class):
                # Add Gaussian noise in tangent space
                noise = np.random.randn(num_dims) * noise_std
                point_tangent = center + noise

                # Convert from tangent space at origin to Poincaré ball
                # Using exponential map approximation
                norm = np.linalg.norm(point_tangent)
                if norm > 0:
                    # Hyperbolic tangent function for mapping
                    r = np.tanh(norm / 2)
                    point_poincare = (r / norm) * point_tangent
                else:
                    point_poincare = point_tangent

                # Ensure point is strictly inside Poincaré ball
                point_norm = np.linalg.norm(point_poincare)
                if point_norm >= 0.99:
                    point_poincare = point_poincare / point_norm * 0.99

                all_points.append(point_poincare)
                all_labels.append(class_idx)

        data_poincare = np.array(all_points)
        labels = np.array(all_labels)

        # Convert from Poincaré to hyperboloid coordinates
        data_hyperboloid = self._poincare_to_hyperboloid(data_poincare)

        # Shuffle
        indices = np.random.permutation(len(data_hyperboloid))
        data_hyperboloid = data_hyperboloid[indices]
        labels = labels[indices]

        return data_hyperboloid, labels

    def _poincare_to_hyperboloid(self, points: np.ndarray) -> np.ndarray:
        """
        Convert points from Poincaré ball to hyperboloid model.

        Parameters:
        -----------
        points : np.ndarray
            Points in Poincaré ball (N, D)

        Returns:
        --------
        hyperboloid_points : np.ndarray
            Points in hyperboloid model (N, D+1)
        """
        # Poincaré to hyperboloid conversion
        norm_sq = np.sum(points**2, axis=1, keepdims=True)

        # Hyperboloid coordinates [t, x1, x2, ..., xn]
        # where t^2 - x1^2 - x2^2 - ... - xn^2 = 1
        t = (1 + norm_sq) / (1 - norm_sq)
        x = (2 * points) / (1 - norm_sq)

        hyperboloid_points = np.concatenate([t, x], axis=1)
        return hyperboloid_points

    def _hyperboloid_to_poincare(self, points: np.ndarray) -> np.ndarray:
        """
        Convert points from hyperboloid model to Poincaré ball.

        Parameters:
        -----------
        points : np.ndarray
            Points in hyperboloid model (N, D+1)

        Returns:
        --------
        poincare_points : np.ndarray
            Points in Poincaré ball (N, D)
        """
        t = points[:, 0:1]
        x = points[:, 1:]

        poincare_points = x / (1 + t)
        return poincare_points

    def generate_embeddings_and_labels(
        self,
        dimension: int = 2,
        num_samples: int = 1250,
        num_classes: int = 4,
        seed: int = 42,
        noise_std: float = 2.0,
        separation: float = 2.0,
        convert_to_poincare: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Gaussian embeddings in hyperbolic space.

        Parameters:
        -----------
        dimension : int
            Dimensionality of the hyperbolic space
        num_samples : int
            Total number of samples to generate
        num_classes : int
            Number of classes/clusters
        seed : int
            Random seed
        noise_std : float
            Standard deviation of cluster noise
        separation : float
            Separation between cluster centers
        convert_to_poincare : bool
            If True, return embeddings in Poincaré ball coordinates
            If False, return in hyperboloid coordinates

        Returns:
        --------
        embeddings : np.ndarray
            Embeddings (N, D) if Poincaré or (N, D+1) if hyperboloid
        labels : np.ndarray
            Labels (N,)
        """
        print(f"Generating {num_samples} Gaussian hyperbolic embeddings...")
        print(f"  Dimension: {dimension}")
        print(f"  Classes: {num_classes}")
        print(f"  Noise std: {noise_std}")
        print(f"  Separation: {separation}")

        # Generate wrapped normal mixture in hyperboloid coordinates
        embeddings, labels = self._wrapped_normal_mixture(
            num_dims=dimension,
            num_classes=num_classes,
            num_points=num_samples,
            seed=seed,
            noise_std=noise_std,
            separation=separation,
        )

        if convert_to_poincare:
            embeddings = self._hyperboloid_to_poincare(embeddings)
            print(f"Generated embeddings in Poincaré ball: {embeddings.shape}")
        else:
            print(f"Generated embeddings in hyperboloid model: {embeddings.shape}")

        print(f"Labels distribution: {np.bincount(labels)}")

        return embeddings, labels

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
        # Try to use the HyperbolicConversions utility if available
        try:
            from utils.geometric_conversions import HyperbolicConversions

            distances = HyperbolicConversions.compute_distances(embeddings, space="poincare")
            return distances
        except ImportError:
            # Fallback: compute Poincaré distances manually
            print("Using manual Poincaré distance computation...")
            return self._compute_poincare_distances_manual(embeddings)

    def _compute_poincare_distances_manual(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Manually compute pairwise Poincaré distances.

        The Poincaré distance is:
        d(x, y) = arcosh(1 + 2 * ||x - y||^2 / ((1 - ||x||^2)(1 - ||y||^2)))

        Parameters:
        -----------
        embeddings : np.ndarray
            Embeddings in Poincaré disk (N, D)

        Returns:
        --------
        distances : np.ndarray
            Pairwise distance matrix (N, N)
        """
        n = len(embeddings)
        distances = np.zeros((n, n))

        # Compute norms
        norms_sq = np.sum(embeddings**2, axis=1)

        for i in range(n):
            for j in range(i + 1, n):
                # ||x - y||^2
                diff_norm_sq = np.sum((embeddings[i] - embeddings[j]) ** 2)

                # Poincaré distance formula
                numerator = 2 * diff_norm_sq
                denominator = (1 - norms_sq[i]) * (1 - norms_sq[j])

                # Avoid numerical issues
                denominator = max(denominator, 1e-10)

                arg = 1 + numerator / denominator
                arg = max(arg, 1.0)  # arcosh requires arg >= 1

                dist = np.arccosh(arg)
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def load_as_networkx(
        self,
        dimension: int = 2,
        num_samples: int = 1250,
        num_classes: int = 2,
        k_neighbors: int = 10,
        seed: int = 42,
        noise_std: float = 2.0,
        separation: float = 2.0,
    ) -> Tuple[nx.Graph, np.ndarray, np.ndarray, dict]:
        """
        Generate Gaussian data and convert to NetworkX graph.

        Parameters:
        -----------
        dimension : int
            Dimensionality of hyperbolic space
        num_samples : int
            Total number of samples
        num_classes : int
            Number of classes
        k_neighbors : int
            Number of neighbors for k-NN graph construction
        seed : int
            Random seed
        noise_std : float
            Standard deviation of cluster noise
        separation : float
            Separation between cluster centers

        Returns:
        --------
        graph : nx.Graph
            NetworkX graph
        labels : np.ndarray
            Node labels
        node_indices : np.ndarray
            Node indices
        metadata : dict
            Dataset metadata
        """
        # Generate embeddings and labels
        embeddings, labels = self.generate_embeddings_and_labels(
            dimension=dimension,
            num_samples=num_samples,
            num_classes=num_classes,
            seed=seed,
            noise_std=noise_std,
            separation=separation,
        )

        # Convert to graph
        graph, labels, node_indices = self.embeddings_to_graph(embeddings, labels, k_neighbors=k_neighbors, distance_metric="poincare")

        # Create metadata
        metadata = {
            "dataset_name": f"gaussian_d{dimension}_c{num_classes}",
            "num_samples": len(labels),
            "num_classes": num_classes,
            "k_neighbors": k_neighbors,
            "embedding_dim": embeddings.shape[1],
            "dimension": dimension,
            "noise_std": noise_std,
            "separation": separation,
        }

        return graph, labels, node_indices, metadata


def load_gaussian_dataset(
    dimension: int = 2,
    num_samples: int = 1000,
    num_classes: int = 2,
    k_neighbors: int = 10,
    seed: int = 42,
    noise_std: float = 2.0,
    separation: float = 2.0,
    data_dir: str = "./data/gaussian",
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Convenience function to load Gaussian hyperbolic dataset.

    Parameters:
    -----------
    dimension : int
        Dimensionality of hyperbolic space
    num_samples : int
        Total number of samples to generate
    num_classes : int
        Number of classes/clusters
    k_neighbors : int
        Number of neighbors for k-NN graph construction
    seed : int
        Random seed
    noise_std : float
        Standard deviation of cluster noise
    separation : float
        Separation between cluster centers
    data_dir : str
        Directory to cache data

    Returns:
    --------
    graph : nx.Graph
        NetworkX graph representation
    labels : np.ndarray
        Node labels
    node_indices : np.ndarray
        Node indices
    """
    loader = GaussianLoader(data_dir=data_dir)

    # Generate dataset
    graph, labels, node_indices, metadata = loader.load_as_networkx(
        dimension=dimension,
        num_samples=num_samples,
        num_classes=num_classes,
        k_neighbors=k_neighbors,
        seed=seed,
        noise_std=noise_std,
        separation=separation,
    )

    print(f"\nGaussian Dataset Generated:")
    print(f"  Dataset: {metadata['dataset_name']}")
    print(f"  Samples: {metadata['num_samples']}")
    print(f"  Classes: {metadata['num_classes']}")
    print(f"  Dimension: {metadata['dimension']}")
    print(f"  Noise std: {metadata['noise_std']}")
    print(f"  Separation: {metadata['separation']}")

    return graph, labels, node_indices
