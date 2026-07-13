"""
SNAP Dataset Loader for Stanford SNAP Datasets

This module provides functionality to download, parse, and load various Stanford SNAP
datasets as NetworkX graphs. Currently supports:
- AS (Autonomous Systems) dataset
- Facebook ego networks dataset

Dataset Information:
- AS: https://snap.stanford.edu/data/as.html
- Facebook: https://snap.stanford.edu/data/egonets-Facebook.html
"""

import gzip
import os
import random
import urllib.request
from typing import Optional, Tuple

import networkx as nx


# Dataset configurations
DATASET_CONFIGS = {
    "as": {
        "url": "https://snap.stanford.edu/data/as20000102.txt.gz",
        "filename": "as20000102.txt.gz",
        "expected_nodes": 6474,
        "expected_edges": 12572,
        "separator": "\t",  # Tab-separated
        "name": "Stanford SNAP AS-20000102",
        "date": "January 2, 2000",
        "source": "University of Oregon Route Views Project",
        "description": "Autonomous Systems graph from BGP routing data",
    },
    "facebook": {
        "url": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        "filename": "facebook_combined.txt.gz",
        "expected_nodes": 4039,
        "expected_edges": 88234,
        "separator": " ",  # Space-separated
        "name": "Stanford SNAP Facebook Ego Networks",
        "date": "2012",
        "source": "Facebook App Survey",
        "description": "Facebook social circles (ego networks) from survey participants",
    },
}


class SNAPDatasetLoader:
    """
    Unified loader for Stanford SNAP datasets.

    This class handles downloading, caching, parsing, and converting SNAP datasets
    into NetworkX graph format with proper validation.
    """

    def __init__(self, cache_dir: str = "data/snap_dataset"):
        """
        Initialize the SNAP dataset loader.

        Args:
            cache_dir: Base directory to cache downloaded dataset files
        """
        self.cache_dir = cache_dir
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

    def _get_dataset_config(self, dataset_type: str) -> dict:
        """
        Get configuration for a specific dataset type.

        Args:
            dataset_type: Type of dataset ("as" or "facebook")

        Returns:
            Dictionary with dataset configuration

        Raises:
            ValueError: If dataset_type is not supported
        """
        if dataset_type not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. Supported types: {list(DATASET_CONFIGS.keys())}")
        return DATASET_CONFIGS[dataset_type]

    def _get_dataset_path(self, dataset_type: str) -> str:
        """
        Get the path where a dataset file should be cached.

        Args:
            dataset_type: Type of dataset ("as" or "facebook")

        Returns:
            Path to the dataset file
        """
        config = self._get_dataset_config(dataset_type)
        dataset_subdir = os.path.join(self.cache_dir, dataset_type)
        os.makedirs(dataset_subdir, exist_ok=True)
        return os.path.join(dataset_subdir, config["filename"])

    def download_dataset(self, dataset_type: str, force_download: bool = False) -> str:
        """
        Download a SNAP dataset if not already cached.

        Args:
            dataset_type: Type of dataset ("as" or "facebook")
            force_download: If True, download even if file exists

        Returns:
            Path to the downloaded dataset file

        Raises:
            Exception: If download fails
        """
        config = self._get_dataset_config(dataset_type)
        dataset_path = self._get_dataset_path(dataset_type)

        if os.path.exists(dataset_path) and not force_download:
            print(f"Dataset already cached at: {dataset_path}")
            return dataset_path

        print(f"Downloading {dataset_type.upper()} dataset from: {config['url']}")
        try:
            urllib.request.urlretrieve(config["url"], dataset_path)
            print(f"Dataset downloaded successfully to: {dataset_path}")
            return dataset_path
        except Exception as e:
            raise Exception(f"Failed to download dataset: {str(e)}")

    def parse_dataset(self, dataset_type: str, file_path: Optional[str] = None) -> nx.Graph:
        """
        Parse a SNAP dataset file and create a NetworkX graph.

        Args:
            dataset_type: Type of dataset ("as" or "facebook")
            file_path: Path to the dataset file (uses cached file if None)

        Returns:
            NetworkX Graph object

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            Exception: If parsing fails
        """
        config = self._get_dataset_config(dataset_type)
        if file_path is None:
            file_path = self._get_dataset_path(dataset_type)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        print(f"Parsing {dataset_type.upper()} dataset from: {file_path}")

        # Create empty graph
        graph = nx.Graph()
        separator = config["separator"]

        try:
            # Open and parse the gzipped file
            with gzip.open(file_path, "rt") as f:
                edge_count = 0
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue

                    # Parse edge using the configured separator
                    parts = line.split(separator)
                    if len(parts) != 2:
                        # Fallback to whitespace splitting if configured separator fails
                        parts = line.split()
                        if len(parts) != 2:
                            print(f"Warning: Skipping malformed line {line_num}: {line}")
                            continue

                    try:
                        node1, node2 = int(parts[0]), int(parts[1])

                        # Skip self-loops
                        if node1 == node2:
                            continue

                        # Add edge to graph
                        graph.add_edge(node1, node2)
                        edge_count += 1

                    except ValueError:
                        print(f"Warning: Invalid node IDs on line {line_num}: {line}")
                        continue

                print(f"Parsed {edge_count} edges from dataset")

        except Exception as e:
            raise Exception(f"Failed to parse dataset: {str(e)}")

        return graph

    def validate_graph(self, graph: nx.Graph, dataset_type: str) -> bool:
        """
        Validate the loaded graph against expected statistics.

        Args:
            graph: NetworkX graph to validate
            dataset_type: Type of dataset ("as" or "facebook")

        Returns:
            True if validation passes, False otherwise
        """
        config = self._get_dataset_config(dataset_type)
        num_nodes = len(graph.nodes())
        num_edges = len(graph.edges())

        print("Graph Statistics:")
        print(f"  Nodes: {num_nodes} (expected: {config['expected_nodes']})")
        print(f"  Edges: {num_edges} (expected: {config['expected_edges']})")

        # Check if statistics match expected values
        nodes_match = num_nodes == config["expected_nodes"]
        edges_match = num_edges == config["expected_edges"]

        if nodes_match and edges_match:
            print("✓ Graph validation passed!")
            return True
        else:
            print("✗ Graph validation failed!")
            if not nodes_match:
                print(f"  Node count mismatch: got {num_nodes}, expected {config['expected_nodes']}")
            if not edges_match:
                print(f"  Edge count mismatch: got {num_edges}, expected {config['expected_edges']}")
            return False

    def get_graph_info(self, graph: nx.Graph) -> dict:
        """
        Get detailed information about the graph.

        Args:
            graph: NetworkX graph

        Returns:
            Dictionary with graph statistics
        """
        info = {
            "nodes": len(graph.nodes()),
            "edges": len(graph.edges()),
            "is_connected": nx.is_connected(graph),
            "number_of_components": nx.number_connected_components(graph),
        }

        if info["is_connected"]:
            info["diameter"] = nx.diameter(graph)
            info["average_clustering"] = nx.average_clustering(graph)
            info["average_shortest_path_length"] = nx.average_shortest_path_length(graph)
        else:
            # Get info for largest connected component
            largest_cc = max(nx.connected_components(graph), key=len)
            largest_subgraph = graph.subgraph(largest_cc)
            info["largest_component_nodes"] = len(largest_cc)
            info["largest_component_edges"] = len(largest_subgraph.edges())
            info["largest_component_diameter"] = nx.diameter(largest_subgraph)
            info["largest_component_avg_clustering"] = nx.average_clustering(largest_subgraph)

        return info

    def load_networkx(self, dataset_type: str, force_download: bool = False, validate: bool = True) -> Tuple[nx.Graph, dict]:
        """
        Complete pipeline to load a SNAP dataset as NetworkX graph.

        Args:
            dataset_type: Type of dataset ("as" or "facebook")
            force_download: If True, re-download dataset even if cached
            validate: If True, validate graph against expected statistics

        Returns:
            Tuple of (NetworkX graph, metadata dictionary)

        Raises:
            Exception: If any step in the pipeline fails
        """
        config = self._get_dataset_config(dataset_type)
        dataset_name = config["name"]

        print(f"Loading {dataset_name}")
        print("=" * len(dataset_name))

        # Download dataset
        dataset_path = self.download_dataset(dataset_type, force_download=force_download)

        # Parse dataset
        graph = self.parse_dataset(dataset_type, dataset_path)

        # Validate if requested
        if validate:
            validation_passed = self.validate_graph(graph, dataset_type)
            if not validation_passed:
                print("Warning: Graph validation failed, but continuing...")

        # Get detailed graph information
        graph_info = self.get_graph_info(graph)

        # Create metadata
        metadata = {
            "dataset_name": dataset_name,
            "dataset_url": config["url"],
            "date": config["date"],
            "source": config["source"],
            "description": config["description"],
            "validation_passed": validate and self.validate_graph(graph, dataset_type),
            **graph_info,
        }

        print("\nDataset loaded successfully!")
        print(f"Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

        return graph, metadata

    def load_as_networkx(self, force_download: bool = False, validate: bool = True) -> Tuple[nx.Graph, dict]:
        """
        Complete pipeline to load AS dataset as NetworkX graph.
        Convenience method for backward compatibility.

        Args:
            force_download: If True, re-download dataset even if cached
            validate: If True, validate graph against expected statistics

        Returns:
            Tuple of (NetworkX graph, metadata dictionary)

        Raises:
            Exception: If any step in the pipeline fails
        """
        return self.load_networkx("as", force_download=force_download, validate=validate)


# Backward compatibility: Create alias for old class name
ASDatasetLoader = SNAPDatasetLoader


def create_subgraph(graph: nx.Graph, target_nodes: int = 1000, random_seed: Optional[int] = None) -> nx.Graph:
    """
    Create a connected subgraph with approximately target_nodes nodes using BFS.

    This function performs a breadth-first search starting from a random node
    and collects nodes until reaching the target number. The resulting subgraph
    will be connected and contain up to target_nodes nodes.

    Args:
        graph: NetworkX graph to sample from
        target_nodes: Target number of nodes (default: 1000)
        random_seed: Random seed for reproducibility (default: None)

    Returns:
        NetworkX Graph subgraph with up to target_nodes nodes. If the original
        graph has fewer than target_nodes nodes, returns the entire graph.
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # If graph is smaller than target, return entire graph
    if len(graph.nodes()) <= target_nodes:
        return graph.copy()

    # Select a random starting node
    start_node = random.choice(list(graph.nodes()))

    # Perform BFS to collect nodes
    visited = set()
    queue = [start_node]
    visited.add(start_node)

    # Continue BFS until we have enough nodes or run out of nodes to visit
    while queue and len(visited) < target_nodes:
        current = queue.pop(0)

        # Add unvisited neighbors to the queue
        for neighbor in graph.neighbors(current):
            if neighbor not in visited and len(visited) < target_nodes:
                visited.add(neighbor)
                queue.append(neighbor)

    # Create subgraph from collected nodes
    subgraph = graph.subgraph(visited).copy()

    return subgraph


def main():
    """
    Example usage of the SNAP dataset loader.
    """
    # Create loader instance
    loader = SNAPDatasetLoader()

    # Load AS dataset
    print("\n" + "=" * 60)
    print("Loading AS Dataset")
    print("=" * 60)
    try:
        graph, metadata = loader.load_networkx("as")

        print("\nDataset Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        print("\nSample edges (first 10):")
        for i, (u, v) in enumerate(list(graph.edges())[:10]):
            print(f"  {u} -- {v}")

        print("\nNode degree statistics:")
        degrees = [graph.degree(n) for n in graph.nodes()]
        print(f"  Min degree: {min(degrees)}")
        print(f"  Max degree: {max(degrees)}")
        print(f"  Average degree: {sum(degrees) / len(degrees):.2f}")

    except Exception as e:
        print(f"Error loading AS dataset: {str(e)}")

    # Load Facebook dataset
    print("\n" + "=" * 60)
    print("Loading Facebook Dataset")
    print("=" * 60)
    try:
        graph, metadata = loader.load_networkx("facebook")

        print("\nDataset Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        print("\nSample edges (first 10):")
        for i, (u, v) in enumerate(list(graph.edges())[:10]):
            print(f"  {u} -- {v}")

        print("\nNode degree statistics:")
        degrees = [graph.degree(n) for n in graph.nodes()]
        print(f"  Min degree: {min(degrees)}")
        print(f"  Max degree: {max(degrees)}")
        print(f"  Average degree: {sum(degrees) / len(degrees):.2f}")

    except Exception as e:
        print(f"Error loading Facebook dataset: {str(e)}")


if __name__ == "__main__":
    main()



