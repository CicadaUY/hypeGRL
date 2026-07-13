"""
General Hyperbolic KNN Classification Script

This script implements K-Nearest Neighbors classification using hyperbolic distance
on various datasets including PoincareMaps datasets, OGB datasets (ogbn-arxiv), etc.
"""

import argparse
import json
import os
import pickle
import random
from datetime import datetime
from typing import Tuple

import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from gaussian_loader import load_gaussian_dataset
from hyperbolic_embeddings import HyperbolicEmbeddings
from neuroseed_loader import NeuroSEEDLoader
from poincare_maps_networkx_loader import PoincareMapsLoader
from utils.geometric_conversions import HyperbolicConversions

# Patch torch.load for PyTorch 2.6+ compatibility with OGB datasets
# This needs to happen at module level before OGB imports torch.load
try:
    import torch

    # Store original torch.load
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        """Patched torch.load that defaults to weights_only=False for OGB compatibility."""
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    # Apply patch
    torch.load = _patched_torch_load

    # Add safe globals for torch_geometric classes if available
    try:
        from torch_geometric.data import Data
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, Data])
    except (ImportError, AttributeError):
        pass

except ImportError:
    # torch not available, will fail later if OGB is used
    pass


def hyperbolic_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Poincaré distance between two points using the function from geometric_conversions.py.

    This wrapper function is compatible with sklearn's metric interface.

    Parameters:
    -----------
    x : np.ndarray
        First point in Poincaré disk (1D array)
    y : np.ndarray
        Second point in Poincaré disk (1D array)

    Returns:
    --------
    float
        Hyperbolic distance between x and y
    """
    # Use compute_distances from geometric_conversions.py
    # Stack the two points into a 2-row array
    points = np.vstack([x, y])
    distance_matrix = HyperbolicConversions.compute_distances(points, space="poincare")
    # Extract the distance between the two points (off-diagonal element)
    return float(distance_matrix[0, 1])


def extract_labels_from_graph(graph: nx.Graph) -> np.ndarray:
    """
    Extract labels from graph node attributes and convert to numeric.

    Parameters:
    -----------
    graph : nx.Graph
        NetworkX graph with node labels

    Returns:
    --------
    np.ndarray
        Numeric labels array
    """
    labels = [graph.nodes[n].get("label", None) for n in graph.nodes()]
    if all(label is None for label in labels):
        raise ValueError("Graph does not contain node labels")
    # Convert string labels to numeric using LabelEncoder
    encoder = LabelEncoder()
    numeric_labels = encoder.fit_transform(labels)
    return numeric_labels


def load_ogb_dataset(
    dataset_name: str = "ogbn-arxiv",
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Load OGB dataset and convert to NetworkX format.

    Parameters:
    -----------
    dataset_name : str
        Name of the OGB dataset (e.g., "ogbn-arxiv")

    Returns:
    --------
    graph : nx.Graph
        NetworkX graph representation (undirected)
    labels : np.ndarray
        Node labels (numeric)
    node_indices : np.ndarray
        Array of node indices
    """
    print(f"Loading OGB dataset: {dataset_name}...")

    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError:
        raise ImportError("ogb package is required. Install it with: pip install ogb")

    dataset = PygNodePropPredDataset(name=dataset_name)
    graph_pyg = dataset[0]  # PyTorch Geometric graph

    print(f"Dataset loaded: {graph_pyg.num_nodes} nodes, {graph_pyg.num_edges} edges")

    # Convert to NetworkX (undirected)
    graph = nx.Graph()
    edge_index = graph_pyg.edge_index.numpy()
    edges = [(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1])]
    graph.add_edges_from(edges)
    graph.add_nodes_from(range(graph_pyg.num_nodes))

    # Extract labels (already numeric)
    labels = graph_pyg.y.numpy().flatten()  # Flatten from (N, 1) to (N,)
    node_indices = np.arange(graph_pyg.num_nodes)

    print(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Number of classes: {len(np.unique(labels))}")
    print(f"Label distribution: {np.bincount(labels)}")

    return graph, labels, node_indices


def load_polblogs_dataset() -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Load PolBlogs dataset from PyTorch Geometric and convert to NetworkX format.

    Returns:
    --------
    graph : nx.Graph
        NetworkX graph representation
    labels : np.ndarray
        Node labels (numeric)
    node_indices : np.ndarray
        Array of node indices
    """
    print("Loading PolBlogs dataset...")

    try:
        from torch_geometric.datasets import PolBlogs
    except ImportError:
        raise ImportError("torch_geometric package is required. Install it with: pip install torch-geometric")

    # Load dataset (it will be downloaded if not present)
    dataset = PolBlogs(root="./data/polblogs")
    data = dataset[0]

    print(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"Number of classes: {dataset.num_classes}")

    # Convert to NetworkX graph
    edge_index = data.edge_index.numpy()
    edges = [(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1])]

    graph = nx.Graph()
    graph.add_edges_from(edges)
    graph.add_nodes_from(range(data.num_nodes))

    # Get labels
    labels = data.y.numpy()

    # Get node indices
    node_indices = np.arange(data.num_nodes)

    print(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Number of classes: {len(np.unique(labels))}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Extract largest connected component (needed for some embedding methods like dmercator)
    if not nx.is_connected(graph):
        print("\nGraph has multiple components. Extracting largest connected component...")
        # Get the largest connected component
        largest_cc = max(nx.connected_components(graph), key=len)

        # Create subgraph with only the largest component
        graph = graph.subgraph(largest_cc).copy()

        # Create mapping from old to new node indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(largest_cc))}

        # Relabel nodes to be sequential starting from 0
        graph = nx.relabel_nodes(graph, old_to_new)

        # Filter labels and node indices to match the largest component
        labels = labels[sorted(largest_cc)]
        node_indices = np.arange(len(largest_cc))

        print(
            f"Largest component: {graph.number_of_nodes()} nodes ({100*len(largest_cc)/data.num_nodes:.1f}% of original), {graph.number_of_edges()} edges"
        )
        print(f"Number of classes: {len(np.unique(labels))}")
        print(f"Label distribution: {np.bincount(labels)}")

    return graph, labels, node_indices


def load_cora_dataset() -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Load CORA dataset and convert to NetworkX format.

    Returns:
    --------
    graph : nx.Graph
        NetworkX graph representation
    labels : np.ndarray
        Node labels (numeric)
    node_indices : np.ndarray
        Array of node indices
    """
    print("Loading CORA dataset...")

    with open("./data/Cora/cora_graph.pkl", "rb") as f:
        edge_list = pickle.load(f)
    with open("./data/Cora/cora_graph.json", "r") as f:
        graph_data = json.load(f)

    # Build networkx graph from edge list
    graph = nx.Graph()
    graph.add_edges_from(edge_list)

    # Get labels
    labels = np.array(graph_data.get("y", []))

    # Get node indices
    node_indices = np.arange(len(graph.nodes()))

    print(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Number of classes: {len(np.unique(labels))}")
    print(f"Label distribution: {np.bincount(labels)}")

    return graph, labels, node_indices


def load_gaussian_hyperbolic_dataset(
    num_samples: int = 1250,
    num_classes: int = 2,
    dimension: int = 2,
    k_neighbors: int = 10,
    seed: int = 42,
    noise_std: float = 2.0,
    separation: float = 2.0,
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Load Gaussian hyperbolic dataset with synthetic embeddings.

    Parameters:
    -----------
    num_samples : int
        Total number of samples to generate
    num_classes : int
        Number of classes/clusters
    dimension : int
        Dimensionality of the hyperbolic space
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
        NetworkX graph representation (k-NN graph)
    labels : np.ndarray
        Node labels (numeric)
    node_indices : np.ndarray
        Array of node indices
    """
    print("Loading Gaussian hyperbolic dataset...")
    print(f"  Samples: {num_samples}")
    print(f"  Classes: {num_classes}")
    print(f"  Dimension: {dimension}")
    print(f"  k-NN neighbors: {k_neighbors}")
    print(f"  Seed: {seed}")

    # Load the Gaussian dataset using the gaussian_loader
    graph, labels, node_indices = load_gaussian_dataset(
        dimension=dimension,
        num_samples=num_samples,
        num_classes=num_classes,
        k_neighbors=k_neighbors,
        seed=seed,
        noise_std=noise_std,
        separation=separation,
    )

    return graph, labels, node_indices


def load_neuroseed_dataset(
    dimension: int = 2,
    num_samples: int = None,
    k_neighbors: int = 10,
    seed: int = 42,
    data_file: str = "data/neuroseed/americangut.h5ad",
    min_label_count: int = 1000,
    taxonomy_level: str = "taxonomy_1",
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Load NeuroSEED dataset and construct a k-NN graph from pre-computed embeddings.

    Note: The pre-computed embeddings are used ONLY to construct the graph structure.
    New embeddings will be trained on this graph structure.

    Parameters:
    -----------
    dimension : int
        Dimensionality of the hyperbolic embeddings (used for loading pre-computed embeddings)
    num_samples : int or None
        Number of samples to randomly subsample from filtered data (None = use all available data)
    k_neighbors : int
        Number of neighbors for k-NN graph construction
    seed : int
        Random seed for reproducibility
    data_file : str
        Path to the h5ad file containing the NeuroSEED dataset
    min_label_count : int
        Minimum number of samples per label to keep
    taxonomy_level : str
        Taxonomy level to use for labels

    Returns:
    --------
    graph : nx.Graph
        NetworkX graph representation (k-NN graph constructed from pre-computed embeddings)
    labels : np.ndarray
        Node labels (numeric)
    node_indices : np.ndarray
        Array of node indices
    """
    print("Loading NeuroSEED dataset...")
    print(f"  Dimension: {dimension}")
    print(f"  Samples: {num_samples or 'all available'}")
    print(f"  k-NN neighbors: {k_neighbors}")
    print(f"  Seed: {seed}")
    print(f"  Data file: {data_file}")

    # Initialize loader
    loader = NeuroSEEDLoader(data_file=data_file)

    # Get all data (not split yet - we'll split it later in the main function)
    X_train, X_test, y_train, y_test = loader.get_data(
        seed=seed,
        dimension=dimension,
        num_samples=num_samples,
        convert_to_poincare=True,
        min_label_count=min_label_count,
        taxonomy_level=taxonomy_level,
    )

    # Combine train and test data back together (we'll split it later in main)
    embeddings = np.vstack([X_train, X_test])
    labels = np.hstack([y_train, y_test])

    print(f"  Total samples loaded: {len(embeddings)}")

    # Convert embeddings to k-NN graph (using pre-computed embeddings only for graph structure)
    graph, labels, node_indices = loader.embeddings_to_graph(
        embeddings=embeddings,
        labels=labels,
        k_neighbors=k_neighbors,
        distance_metric="poincare",
    )

    print("Note: Graph structure created from pre-computed embeddings.")
    print("      New embeddings will be trained on this graph structure.")

    return graph, labels, node_indices


def load_dataset(
    dataset_name: str,
    dataset_type: str,
    k_neighbors: int = 10,
    datasets_path: str = None,
    num_samples: int = None,
    num_classes: int = 6,
    dimension: int = 2,
    seed: int = 42,
    neuroseed_data_file: str = "data/neuroseed/americangut.h5ad",
    neuroseed_min_label_count: int = 1000,
    neuroseed_taxonomy_level: str = "taxonomy_1",
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Load dataset based on type and return graph, labels, and node indices.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    dataset_type : str
        Type of dataset ("AS", "ogbn-arxiv", "gaussian", "neuroseed", or PoincareMaps dataset name)
    k_neighbors : int
        Number of neighbors for PoincareMaps/Gaussian/NeuroSEED KNN graph construction
    datasets_path : str
        Path to PoincareMaps datasets directory
    num_samples : int or None
        Number of samples for Gaussian dataset / NeuroSEED (None = use all available for NeuroSEED)
    num_classes : int
        Number of classes for Gaussian dataset (only used when dataset_type="gaussian")
    dimension : int
        Dimension for Gaussian/NeuroSEED dataset
    seed : int
        Random seed for Gaussian/NeuroSEED dataset

    Returns:
    --------
    graph : nx.Graph
        NetworkX graph representation
    labels : np.ndarray
        Node labels (numeric)
    node_indices : np.ndarray
        Array of node indices
    """
    if dataset_type == "AS":
        raise ValueError("AS dataset does not contain node labels for classification")
    elif dataset_type == "ogbn-arxiv":
        return load_ogb_dataset("ogbn-arxiv")
    elif dataset_type == "polblogs":
        return load_polblogs_dataset()
    elif dataset_type == "Cora":
        return load_cora_dataset()
    elif dataset_type == "gaussian":
        return load_gaussian_hyperbolic_dataset(
            num_samples=num_samples,
            num_classes=num_classes,
            dimension=dimension,
            k_neighbors=k_neighbors,
            seed=seed,
        )
    elif dataset_type == "neuroseed":
        # NeuroSEED: use pre-computed embeddings only to construct graph, then train new embeddings
        # If num_samples is None, uses all available filtered data (80/20 train/test split)
        return load_neuroseed_dataset(
            dimension=dimension,
            num_samples=num_samples,
            k_neighbors=k_neighbors,
            seed=seed,
            data_file=neuroseed_data_file,
            min_label_count=neuroseed_min_label_count,
            taxonomy_level=neuroseed_taxonomy_level,
        )
    else:
        # PoincareMaps dataset
        if datasets_path is None:
            datasets_path = "models/PoincareMaps/datasets/"

        print(f"Loading PoincareMaps dataset: {dataset_name}...")
        loader = PoincareMapsLoader(datasets_path)
        graph, metadata = loader.load_as_networkx(dataset_name, k_neighbors=k_neighbors)
        labels = extract_labels_from_graph(graph)
        node_indices = np.arange(len(graph.nodes()))

        print(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Number of classes: {len(np.unique(labels))}")
        print(f"Label distribution: {np.bincount(labels)}")

        return graph, labels, node_indices


def train_hyperbolic_embeddings(
    graph: nx.Graph,
    embedding_type: str = "poincare_embeddings",
    model_dir: str = "saved_models/default",
    dim: int = 2,
    random_seed: int = None,
) -> Tuple[np.ndarray, str]:
    """
    Train hyperbolic embeddings for the graph with dataset size-based configurations.

    Parameters:
    -----------
    graph : nx.Graph
        Input graph
    embedding_type : str
        Type of embedding model to use
    model_dir : str
        Directory to save the model
    dim : int
        Embedding dimension
    random_seed : int
        Random seed for reproducibility (default: None)

    Returns:
    --------
    embeddings : np.ndarray
        Node embeddings
    embedding_space : str
        Native embedding space of the model
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
        try:
            import torch

            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)
        except ImportError:
            pass

    print(f"\nTraining {embedding_type} embeddings (seed={random_seed})...")

    # Ensure graph nodes are 0-based consecutive (critical for adjacency matrix creation)
    graph_nodes = sorted(graph.nodes())
    expected_nodes = list(range(len(graph_nodes)))
    if graph_nodes != expected_nodes:
        print("Remapping graph nodes to 0-based consecutive indices...")
        # Create node mapping
        node_mapping = {old_node: new_idx for new_idx, old_node in enumerate(graph_nodes)}
        # Create new graph with remapped nodes
        graph_remapped = nx.Graph()
        for old_node in graph_nodes:
            graph_remapped.add_node(node_mapping[old_node])
        for old_u, old_v in graph.edges():
            if old_u in node_mapping and old_v in node_mapping:
                graph_remapped.add_edge(node_mapping[old_u], node_mapping[old_v])
        graph = graph_remapped
        print(f"Graph remapped: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Prepare edge list
    edge_list = list(graph.edges())
    num_nodes = graph.number_of_nodes()

    # Determine dataset size category
    is_large_dataset = num_nodes >= 100000  # ogbn-arxiv scale
    is_medium_dataset = num_nodes >= 1000

    # Embedding configurations based on dataset size
    if is_large_dataset:
        # Large datasets (>= 100K nodes, e.g., ogbn-arxiv)
        configurations = {
            "poincare_embeddings": {"dim": dim, "negs": 10, "epochs": 500, "batch_size": 512, "dimension": 1},
            "lorentz": {"dim": dim, "epochs": 5000, "batch_size": 2048, "num_nodes": num_nodes},
            "dmercator": {"dim": dim},
            "hydra": {"dim": 3},
            "poincare_maps": {"dim": dim, "epochs": 500},
            "hypermap": {"dim": 3},
            "hydra_plus": {"dim": 3},
        }
    elif is_medium_dataset:
        # Medium datasets (1K-100K nodes)
        configurations = {
            "poincare_embeddings": {"dim": dim, "negs": 5, "epochs": 1000, "batch_size": 256, "dimension": 1},
            "lorentz": {"dim": dim, "epochs": 10000, "batch_size": 1024, "num_nodes": num_nodes},
            "dmercator": {"dim": dim},
            "hydra": {"dim": 2},
            "poincare_maps": {"dim": dim, "epochs": 1000},
            "hypermap": {"dim": 3},
            "hydra_plus": {"dim": 2},
        }
    else:
        # Small datasets (< 1K nodes)
        configurations = {
            "poincare_embeddings": {"dim": dim, "negs": 5, "epochs": 1000, "batch_size": 256, "dimension": 1},
            "lorentz": {"dim": dim, "epochs": 10000, "batch_size": 1024, "num_nodes": num_nodes},
            "dmercator": {"dim": dim},
            "hydra": {"dim": 2},
            "poincare_maps": {"dim": dim, "epochs": 1000},
            "hypermap": {"dim": 3},
            "hydra_plus": {"dim": 2},
        }

    config = configurations.get(embedding_type, configurations["poincare_embeddings"])

    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{embedding_type}_embeddings.bin")

    # Initialize and train embeddings
    embedding_runner = HyperbolicEmbeddings(embedding_type=embedding_type, config=config)

    # Get adjacency matrix for models that need it
    if embedding_type in ["hydra", "poincare_maps", "lorentz", "hydra_plus"]:
        A = nx.to_numpy_array(graph)
        embedding_runner.train(adjacency_matrix=A, model_path=model_path)
    else:
        embedding_runner.train(edge_list=edge_list, model_path=model_path)

    # Get embeddings
    embeddings = embedding_runner.get_all_embeddings(model_path)
    embedding_space = embedding_runner.model.native_space

    print(f"Embeddings trained: shape {embeddings.shape}, space: {embedding_space}")

    # Convert embeddings to Poincaré if needed
    if embedding_space != "poincare":
        print(f"Converting embeddings from {embedding_space} to Poincaré...")
        embeddings = HyperbolicConversions.convert_coordinates(embeddings, embedding_space, "poincare")
        embedding_space = "poincare"

    return embeddings, embedding_space


def evaluate_knn(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k_values: list = [3, 5, 7, 10],
) -> dict:
    """
    Evaluate KNN classification with different k values using hyperbolic distance.

    This function performs a single evaluation pass. For multiple iterations,
    call this function multiple times with different embeddings.

    Parameters:
    -----------
    X_train : np.ndarray
        Training embeddings
    X_test : np.ndarray
        Test embeddings
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Test labels
    k_values : list
        List of k values to test

    Returns:
    --------
    results : dict
        Dictionary containing results for each k value
    """
    results = {}

    for k in k_values:
        # Create KNN classifier with hyperbolic distance
        knn = KNeighborsClassifier(n_neighbors=k, metric=hyperbolic_distance, algorithm="brute")

        # Fit and predict
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[k] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="General Hyperbolic KNN Classification")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ToggleSwitch",
        choices=[
            "AS",
            "ToggleSwitch",
            "Olsson",
            "MyeloidProgenitors",
            "krumsiek11_blobs",
            "Paul",
            "ogbn-arxiv",
            "polblogs",
            "Cora",
            "gaussian",
            "neuroseed",
        ],
        help="Dataset to use",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="hydra_plus",
        choices=["poincare_embeddings", "lorentz", "dmercator", "hydra", "poincare_maps", "hypermap", "hydra_plus"],
        help="Type of embedding model to use.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory to save the trained model (default: saved_models/{dataset})",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing.",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[3, 5, 7, 10],
        help="List of k values to test for KNN.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for train/test split.",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=20,
        help="Number of neighbors for PoincareMaps KNN graph construction (ignored for OGB datasets).",
    )
    parser.add_argument(
        "--datasets_path",
        type=str,
        default="models/PoincareMaps/datasets/",
        help="Path to PoincareMaps datasets directory.",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=10,
        help="Number of iterations to run for computing mean and std of metrics (default: 10).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save results (default: test/other/results/{dataset}/{embedding_type}_knn_results.txt).",
    )
    parser.add_argument(
        "--gaussian_num_samples",
        type=int,
        default=None,
        help="Number of samples for Gaussian/NeuroSEED dataset (None = use all available for NeuroSEED, defaults to 1250 for Gaussian).",
    )
    parser.add_argument(
        "--gaussian_num_classes",
        type=int,
        default=6,
        help="Number of classes for Gaussian dataset (only used when dataset=gaussian).",
    )
    parser.add_argument(
        "--gaussian_dimension",
        type=int,
        default=2,
        help="Dimension for Gaussian dataset (only used when dataset=gaussian).",
    )
    parser.add_argument(
        "--gaussian_seed",
        type=int,
        default=42,
        help="Random seed for Gaussian dataset generation (only used when dataset=gaussian).",
    )
    parser.add_argument(
        "--neuroseed_data_file",
        type=str,
        default="data/neuroseed/americangut.h5ad",
        help="Path to the NeuroSEED h5ad data file (only used when dataset=neuroseed).",
    )
    parser.add_argument(
        "--neuroseed_min_label_count",
        type=int,
        default=1000,
        help="Minimum number of samples per label to keep (only used when dataset=neuroseed).",
    )
    parser.add_argument(
        "--neuroseed_taxonomy_level",
        type=str,
        default="taxonomy_1",
        help="Taxonomy level to use for labels (only used when dataset=neuroseed).",
    )

    args = parser.parse_args()

    # Set default model directory if not provided
    if args.model_dir is None:
        args.model_dir = f"saved_models/{args.dataset}"

    # Load dataset
    try:
        # Use args.dim for NeuroSEED dimension
        # Use args.gaussian_num_samples for both Gaussian and NeuroSEED num_samples
        dimension_to_use = args.dim if args.dataset == "neuroseed" else args.gaussian_dimension

        # Set default num_samples for Gaussian dataset if None
        num_samples_to_use = args.gaussian_num_samples
        if num_samples_to_use is None and args.dataset == "gaussian":
            num_samples_to_use = 1250  # Default for Gaussian

        graph, labels, node_indices = load_dataset(
            dataset_name=args.dataset,
            dataset_type=args.dataset,
            k_neighbors=args.k_neighbors,
            datasets_path=args.datasets_path,
            num_samples=num_samples_to_use,  # Used for both Gaussian and NeuroSEED
            num_classes=args.gaussian_num_classes,
            dimension=dimension_to_use,
            seed=args.gaussian_seed,
            neuroseed_data_file=args.neuroseed_data_file,
            neuroseed_min_label_count=args.neuroseed_min_label_count,
            neuroseed_taxonomy_level=args.neuroseed_taxonomy_level,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Split node indices into train and test sets BEFORE computing embeddings
    print(f"\nSplitting node indices into train/test sets (test_size={args.test_size})...")
    train_indices, test_indices, y_train, y_test = train_test_split(
        node_indices,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    print(f"Training set: {len(train_indices)} samples")
    print(f"Test set: {len(test_indices)} samples")

    # Print number of classes and test set size
    print(f"Number of test nodes: {len(test_indices)}")

    # Evaluate KNN with different k values across multiple iterations
    print(f"\n{'=' * 90}")
    print(f"Running {args.n_iterations} iteration(s) with different embedding seeds")
    print(f"{'=' * 90}")

    # Store results across all iterations
    all_results = {k: {"accuracies": [], "precisions": [], "recalls": [], "f1_scores": []} for k in args.k_values}

    for iteration in range(args.n_iterations):
        print(f"\n{'*' * 90}")
        print(f"ITERATION {iteration + 1}/{args.n_iterations}")
        print(f"{'*' * 90}")

        # Generate embeddings with different random seed for each iteration
        iteration_seed = args.random_state + iteration

        # Train embeddings on the full graph with different seed for each iteration
        print(f"\nTraining embeddings with seed={iteration_seed}...")
        embeddings_iter, _ = train_hyperbolic_embeddings(
            graph=graph,
            embedding_type=args.embedding_type,
            model_dir=os.path.join(args.model_dir, f"iter{iteration}"),
            dim=args.dim,
            random_seed=iteration_seed,
        )

        # Use the pre-determined train/test split indices
        X_train_iter = embeddings_iter[train_indices]
        X_test_iter = embeddings_iter[test_indices]

        # Evaluate KNN for this iteration
        print(f"\nEvaluating KNN for iteration {iteration + 1}...")
        iter_results = evaluate_knn(X_train_iter, X_test_iter, y_train, y_test, k_values=args.k_values)

        # Store results
        for k in args.k_values:
            all_results[k]["accuracies"].append(iter_results[k]["accuracy"])
            all_results[k]["precisions"].append(iter_results[k]["precision"])
            all_results[k]["recalls"].append(iter_results[k]["recall"])
            all_results[k]["f1_scores"].append(iter_results[k]["f1"])

        # Print iteration results
        if args.n_iterations > 1:
            print(f"\nIteration {iteration + 1} Results:")
            print(f"{'k':<5} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
            print("-" * 90)
            for k in sorted(args.k_values):
                r = iter_results[k]
                print(f"{k:<5} " f"{r['accuracy']:<12.4f} " f"{r['precision']:<12.4f} " f"{r['recall']:<12.4f} " f"{r['f1']:<12.4f}")

    # Compute mean and std across iterations
    final_results = {}
    for k in args.k_values:
        final_results[k] = {
            "accuracy_mean": np.mean(all_results[k]["accuracies"]),
            "accuracy_std": np.std(all_results[k]["accuracies"]),
            "precision_mean": np.mean(all_results[k]["precisions"]),
            "precision_std": np.std(all_results[k]["precisions"]),
            "recall_mean": np.mean(all_results[k]["recalls"]),
            "recall_std": np.std(all_results[k]["recalls"]),
            "f1_mean": np.mean(all_results[k]["f1_scores"]),
            "f1_std": np.std(all_results[k]["f1_scores"]),
        }

    # Print summary
    print("\n" + "=" * 90)
    print(f"SUMMARY OF RESULTS (Mean ± Std over {args.n_iterations} iteration(s))")
    print("=" * 90)
    if args.n_iterations > 1:
        print(f"{'k':<5} {'Accuracy':<20} {'Precision':<20} {'Recall':<20} {'F1-Score':<20}")
        print("-" * 90)
        for k in sorted(final_results.keys()):
            r = final_results[k]
            print(
                f"{k:<5} "
                f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}  "
                f"{r['precision_mean']:.4f} ± {r['precision_std']:.4f}  "
                f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f}  "
                f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
            )
    else:
        print(f"{'k':<5} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 90)
        for k in sorted(final_results.keys()):
            r = final_results[k]
            print(
                f"{k:<5} "
                f"{r['accuracy_mean']:<12.4f} "
                f"{r['precision_mean']:<12.4f} "
                f"{r['recall_mean']:<12.4f} "
                f"{r['f1_mean']:<12.4f}"
            )
    print("=" * 90)

    # Find best k
    best_k = max(final_results.keys(), key=lambda k: final_results[k]["accuracy_mean"])
    print(
        f"\nBest k value: {best_k} (Accuracy: {final_results[best_k]['accuracy_mean']:.4f} ± {final_results[best_k]['accuracy_std']:.4f})"
    )

    # Save results to file
    if args.output_file is None:
        # Create default output path
        output_dir = f"test/other/results/{args.dataset}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{args.embedding_type}_knn_results.txt")
    else:
        output_file = args.output_file
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        # Write header with experiment configuration
        f.write("=" * 90 + "\n")
        f.write("HYPERBOLIC KNN CLASSIFICATION RESULTS\n")
        f.write("=" * 90 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("CONFIGURATION:\n")
        f.write("-" * 90 + "\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Embedding Type: {args.embedding_type}\n")
        f.write(f"Embedding Dimension: {args.dim}\n")
        f.write(f"Test Size: {args.test_size}\n")
        f.write(f"Random State: {args.random_state}\n")
        f.write(f"Number of Iterations: {args.n_iterations}\n")
        f.write(f"K Values: {args.k_values}\n")
        f.write(f"K Neighbors (graph construction): {args.k_neighbors}\n")
        f.write(f"Number of Graph Nodes: {graph.number_of_nodes()}\n")
        f.write(f"Number of Graph Edges: {graph.number_of_edges()}\n")
        f.write(f"Number of Classes: {len(np.unique(labels))}\n")
        f.write(f"Training Set Size: {len(train_indices)}\n")
        f.write(f"Test Set Size: {len(test_indices)}\n")
        f.write("\n")

        # Write summary results
        f.write("=" * 90 + "\n")
        f.write(f"SUMMARY OF RESULTS (Mean ± Std over {args.n_iterations} iteration(s))\n")
        f.write("=" * 90 + "\n")

        if args.n_iterations > 1:
            f.write(f"{'k':<5} {'Accuracy':<20} {'Precision':<20} {'Recall':<20} {'F1-Score':<20}\n")
            f.write("-" * 90 + "\n")
            for k in sorted(final_results.keys()):
                r = final_results[k]
                f.write(
                    f"{k:<5} "
                    f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}  "
                    f"{r['precision_mean']:.4f} ± {r['precision_std']:.4f}  "
                    f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f}  "
                    f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}\n"
                )
        else:
            f.write(f"{'k':<5} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
            f.write("-" * 90 + "\n")
            for k in sorted(final_results.keys()):
                r = final_results[k]
                f.write(
                    f"{k:<5} "
                    f"{r['accuracy_mean']:<12.4f} "
                    f"{r['precision_mean']:<12.4f} "
                    f"{r['recall_mean']:<12.4f} "
                    f"{r['f1_mean']:<12.4f}\n"
                )
        f.write("=" * 90 + "\n\n")

        # Write best k
        f.write(
            f"Best k value: {best_k} (Accuracy: {final_results[best_k]['accuracy_mean']:.4f} ± {final_results[best_k]['accuracy_std']:.4f})\n\n"
        )

        # Write detailed iteration results if multiple iterations
        if args.n_iterations > 1:
            f.write("=" * 90 + "\n")
            f.write("DETAILED ITERATION RESULTS\n")
            f.write("=" * 90 + "\n")
            for k in sorted(args.k_values):
                f.write(f"\nk = {k}:\n")
                f.write(f"  Accuracies:  {all_results[k]['accuracies']}\n")
                f.write(f"  Precisions:  {all_results[k]['precisions']}\n")
                f.write(f"  Recalls:     {all_results[k]['recalls']}\n")
                f.write(f"  F1-Scores:   {all_results[k]['f1_scores']}\n")

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
