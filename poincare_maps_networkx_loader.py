import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csgraph
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph


class PoincareMapsLoader:
    """
    A class to load and process PoincareMaps datasets as NetworkX graphs.

    This class replicates the data loading and graph construction logic
    from the original PoincareMaps repository.
    """

    def __init__(self, datasets_path: str = "datasets/"):
        """
        Initialize the loader with the path to datasets.

        Args:
            datasets_path: Path to the directory containing CSV datasets
        """
        self.datasets_path = Path(datasets_path)
        self.available_datasets = self._discover_datasets()

    def _discover_datasets(self) -> List[str]:
        """Discover available datasets in the datasets directory."""
        if not self.datasets_path.exists():
            warnings.warn(f"Datasets path {self.datasets_path} does not exist")
            return []

        csv_files = list(self.datasets_path.glob("*.csv"))
        return [f.stem for f in csv_files if f.stat().st_size > 0]

    def list_datasets(self) -> List[str]:
        """Return a list of available datasets."""
        return self.available_datasets.copy()

    def load_dataset_info(self) -> Dict[str, Dict]:
        """
        Load information about all available datasets.

        Returns:
            Dictionary with dataset names as keys and info as values
        """
        info = {}
        for dataset_name in self.available_datasets:
            try:
                df = pd.read_csv(self.datasets_path / f"{dataset_name}.csv")
                n_samples, n_features = df.shape
                if "labels" in df.columns:
                    n_features -= 1
                    unique_labels = df["labels"].nunique()
                    label_counts = df["labels"].value_counts().to_dict()
                else:
                    unique_labels = 0
                    label_counts = {}

                file_size = (self.datasets_path / f"{dataset_name}.csv").stat().st_size
                info[dataset_name] = {
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "n_unique_labels": unique_labels,
                    "label_counts": label_counts,
                    "file_size_mb": file_size / (1024 * 1024),
                }
            except Exception as e:
                warnings.warn(f"Could not load info for {dataset_name}: {e}")

        return info

    def prepare_data(
        self,
        dataset_name: str,
        with_labels: bool = True,
        normalize: bool = False,
        n_pca: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess a dataset.

        Args:
            dataset_name: Name of the dataset to load
            with_labels: Whether to extract labels from the last column
            normalize: Whether to apply z-score normalization
            n_pca: Number of PCA components (0 = no PCA, 1 = all components)

        Returns:
            Tuple of (features, labels)
        """
        file_path = self.datasets_path / f"{dataset_name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found")

        df = pd.read_csv(file_path, sep=",")
        n = len(df.columns)

        if with_labels and "labels" in df.columns:
            # Labels are in the 'labels' column
            feature_cols = [col for col in df.columns if col != "labels"]
            x = np.double(df[feature_cols].values)
            labels = df["labels"].values.astype(str)
        elif with_labels:
            # Assume labels are in the last column
            x = np.double(df.values[:, 0 : (n - 1)])
            labels = df.values[:, n - 1].astype(str)
        else:
            x = np.double(df.values)
            labels = np.array(["unknown"] * x.shape[0])

        # Remove features with zero standard deviation
        idx = np.where(np.std(x, axis=0) != 0)[0]
        x = x[:, idx]

        if normalize:
            s = np.std(x, axis=0)
            s[s == 0] = 1
            x = (x - np.mean(x, axis=0)) / s

        if n_pca:
            if n_pca == 1:
                n_pca = x.shape[1]

            nc = min(n_pca, x.shape[1])
            pca = PCA(n_components=nc)
            x = pca.fit_transform(x)

        return x, labels

    def create_knn_graph(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        k_neighbors: int = 15,
        distance_metric: str = "minkowski",
        distance_function: str = "sym",
        connected: bool = True,
    ) -> nx.Graph:
        """
        Create a k-nearest neighbor graph from features.

        Args:
            features: Feature matrix (n_samples x n_features)
            labels: Sample labels
            k_neighbors: Number of nearest neighbors
            distance_metric: Distance metric for KNN
            distance_function: How to combine directed edges ('sym', 'min')
            connected: Whether to force the graph to be connected

        Returns:
            NetworkX graph
        """
        # Create KNN graph using scikit-learn
        knn_matrix = kneighbors_graph(
            features,
            k_neighbors,
            mode="distance",
            metric=distance_metric,
            include_self=False,
        ).toarray()

        # Make symmetric or take minimum
        if "sym" in distance_function.lower():
            knn_matrix = np.maximum(knn_matrix, knn_matrix.T)
        else:
            knn_matrix = np.minimum(knn_matrix, knn_matrix.T)

        # Check connectivity and connect if needed
        n_components, component_labels = csgraph.connected_components(knn_matrix)

        if connected and n_components > 1:
            knn_matrix = self._connect_components(knn_matrix, features, component_labels, distance_metric)

        # Create NetworkX graph
        G = nx.Graph()

        # Add nodes with attributes
        for i, label in enumerate(labels):
            G.add_node(i, label=label, features=features[i])

        # Add edges with weights
        rows, cols = np.nonzero(knn_matrix)
        for i, j in zip(rows, cols):
            if i < j:  # Avoid duplicate edges in undirected graph
                weight = knn_matrix[i, j]
                if weight > 0:
                    G.add_edge(i, j, weight=weight, distance=weight)

        return G

    def _connect_components(
        self,
        knn_matrix: np.ndarray,
        features: np.ndarray,
        component_labels: np.ndarray,
        distance_metric: str,
    ) -> np.ndarray:
        """
        Connect disconnected components by adding edges between closest nodes.

        This replicates the connect_knn function from the original code.
        """
        distances = pairwise_distances(features, metric=distance_metric)
        n_components = len(np.unique(component_labels))

        cur_comp = 0
        while n_components > 1:
            idx_cur = np.where(component_labels == cur_comp)[0]
            idx_rest = np.where(component_labels != cur_comp)[0]
            d = distances[idx_cur][:, idx_rest]
            ia, ja = np.where(d == np.min(d))
            i, j = ia[0], ja[0]

            # Add edge between closest nodes from different components
            node_i = idx_cur[i]
            node_j = idx_rest[j]
            knn_matrix[node_i, node_j] = distances[node_i, node_j]
            knn_matrix[node_j, node_i] = distances[node_j, node_i]

            # Merge components
            nearest_comp = component_labels[node_j]
            component_labels[component_labels == nearest_comp] = cur_comp
            n_components -= 1

        return knn_matrix

    def load_as_networkx(
        self,
        dataset_name: str,
        k_neighbors: int = 15,
        distance_metric: str = "minkowski",
        normalize: bool = False,
        n_pca: int = 0,
        connected: bool = True,
    ) -> Tuple[nx.Graph, Dict]:
        """
        Load a dataset as a NetworkX graph with default parameters.

        Args:
            dataset_name: Name of the dataset
            k_neighbors: Number of nearest neighbors for graph construction
            distance_metric: Distance metric for KNN
            normalize: Whether to normalize features
            n_pca: Number of PCA components (0 = no PCA)
            connected: Whether to ensure graph connectivity

        Returns:
            Tuple of (NetworkX graph, metadata dictionary)
        """
        # Load and preprocess data
        features, labels = self.prepare_data(dataset_name, with_labels=True, normalize=normalize, n_pca=n_pca)

        # Create graph
        G = self.create_knn_graph(
            features,
            labels,
            k_neighbors=k_neighbors,
            distance_metric=distance_metric,
            connected=connected,
        )

        # Create metadata
        metadata = {
            "dataset_name": dataset_name,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "n_features": features.shape[1],
            "k_neighbors": k_neighbors,
            "distance_metric": distance_metric,
            "normalized": normalize,
            "n_pca_components": n_pca if n_pca > 0 else None,
            "connected": connected,
            "unique_labels": list(np.unique(labels)),
            "is_connected": nx.is_connected(G),
        }

        return G, metadata

    def visualize_graph_stats(self, G: nx.Graph, metadata: Dict, figsize: Tuple[int, int] = (15, 10)):
        """
        Create visualizations of graph statistics.

        Args:
            G: NetworkX graph
            metadata: Metadata dictionary
            figsize: Figure size for plots
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f"Graph Statistics: {metadata['dataset_name']}", fontsize=16)

        # Degree distribution
        degrees = [G.degree(n) for n in G.nodes()]
        axes[0, 0].hist(degrees, bins=30, alpha=0.7, edgecolor="black")
        axes[0, 0].set_title("Degree Distribution")
        axes[0, 0].set_xlabel("Degree")
        axes[0, 0].set_ylabel("Frequency")

        # Edge weight distribution
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        axes[0, 1].hist(weights, bins=30, alpha=0.7, edgecolor="black")
        axes[0, 1].set_title("Edge Weight Distribution")
        axes[0, 1].set_xlabel("Weight (Distance)")
        axes[0, 1].set_ylabel("Frequency")

        # Label distribution
        labels = [G.nodes[n]["label"] for n in G.nodes()]
        label_counts = pd.Series(labels).value_counts()
        axes[0, 2].bar(range(len(label_counts)), label_counts.values)
        axes[0, 2].set_title("Label Distribution")
        axes[0, 2].set_xlabel("Label Index")
        axes[0, 2].set_ylabel("Count")
        axes[0, 2].tick_params(axis="x", rotation=45)

        # Clustering coefficient distribution
        clustering = list(nx.clustering(G).values())
        axes[1, 0].hist(clustering, bins=30, alpha=0.7, edgecolor="black")
        axes[1, 0].set_title("Clustering Coefficient Distribution")
        axes[1, 0].set_xlabel("Clustering Coefficient")
        axes[1, 0].set_ylabel("Frequency")

        # Connected components
        components = list(nx.connected_components(G))
        comp_sizes = [len(comp) for comp in components]
        axes[1, 1].bar(range(len(comp_sizes)), sorted(comp_sizes, reverse=True))
        axes[1, 1].set_title("Connected Component Sizes")
        axes[1, 1].set_xlabel("Component Index")
        axes[1, 1].set_ylabel("Size")

        # Graph metrics summary
        metrics_text = f"""
        Nodes: {G.number_of_nodes()}
        Edges: {G.number_of_edges()}
        Avg Degree: {np.mean(degrees):.2f}
        Density: {nx.density(G):.4f}
        Connected: {nx.is_connected(G)}
        Components: {len(components)}
        Avg Clustering: {nx.average_clustering(G):.3f}
        """
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment="center")
        axes[1, 2].set_title("Graph Metrics")
        axes[1, 2].axis("off")

        plt.tight_layout()
        return fig


def main():
    """Example usage of the PoincareMapsLoader."""
    # Initialize loader - adjust path as needed
    datasets_path = "hyperbolic-embeddings/models/PoincareMaps/datasets/"
    loader = PoincareMapsLoader(datasets_path)

    print("Available datasets:")
    datasets = loader.list_datasets()
    for dataset in datasets:
        print(f"  - {dataset}")

    if not datasets:
        print("No datasets found. Please check the path.")
        return

    # Load dataset info
    print("\nDataset Information:")
    info = loader.load_dataset_info()
    for name, details in info.items():
        print(f"\n{name}:")
        print(f"  Samples: {details['n_samples']}")
        print(f"  Features: {details['n_features']}")
        print(f"  Unique Labels: {details['n_unique_labels']}")
        print(f"  File Size: {details['file_size_mb']:.2f} MB")

    # Load a dataset as NetworkX graph
    if datasets:
        dataset_name = datasets[0]  # Use first available dataset
        print(f"\nLoading {dataset_name} as NetworkX graph...")

        try:
            G, metadata = loader.load_as_networkx(
                dataset_name,
                k_neighbors=15,
                distance_metric="minkowski",
                connected=True,
            )

            print("Graph created successfully!")
            print(f"  Nodes: {G.number_of_nodes()}")
            print(f"  Edges: {G.number_of_edges()}")
            print(f"  Connected: {nx.is_connected(G)}")
            print(f"  Unique labels: {len(metadata['unique_labels'])}")

            # Visualize graph statistics
            loader.visualize_graph_stats(G, metadata)
            plt.savefig(f"{dataset_name}_graph_stats.png", dpi=300, bbox_inches="tight")
            print(f"Graph statistics saved as {dataset_name}_graph_stats.png")

        except Exception as e:
            print(f"Error loading dataset: {e}")


if __name__ == "__main__":
    main()
