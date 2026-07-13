import argparse
import json
import pickle
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from hyperbolic_embeddings import HyperbolicEmbeddings
from utils.geometric_conversions import compute_distances, convert_coordinates


def create_test_tree_graph(branching_factor: int = 2, depth: int = 4) -> nx.Graph:
    """Create a balanced tree graph for testing."""
    return nx.balanced_tree(branching_factor, depth)


def simulate_link_removal(graph: nx.Graph, q: float, seed: int = None) -> Dict[str, List[Tuple[int, int]]]:
    """
    Simulate random link removal experiment.

    Args:
        graph: Input NetworkX graph
        q: Probability of keeping a link (removal probability is 1-q)
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
        - 'omega_E': List of remaining links (existing edges after removal)
        - 'omega_R': List of removed links (missing links)
        - 'omega_N': List of true non-links (pairs that were never connected)
        - 'omega_R_plus_N': Combined list of removed links and true non-links
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Get all edges in the original graph
    original_edges = list(graph.edges())
    n_nodes = len(graph.nodes())

    # Generate all possible node pairs (excluding self-loops)
    all_possible_pairs = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            all_possible_pairs.append((i, j))

    # Convert original edges to set for faster lookup
    original_edges_set = set(original_edges)
    original_edges_set.update([(v, u) for u, v in original_edges])  # Add both directions

    # Identify true non-links (pairs that were never connected)
    omega_N = []
    for pair in all_possible_pairs:
        if pair not in original_edges_set and (pair[1], pair[0]) not in original_edges_set:
            omega_N.append(pair)

    # Simulate link removal with probability 1-q
    omega_E = []  # Remaining links
    omega_R = []  # Removed links

    for edge in original_edges:
        if random.random() < q:
            # Keep the link
            omega_E.append(edge)
        else:
            # Remove the link
            omega_R.append(edge)

    # Combine removed links and true non-links
    omega_R_plus_N = omega_R + omega_N

    return {"omega_E": omega_E, "omega_R": omega_R, "omega_N": omega_N, "omega_R_plus_N": omega_R_plus_N}


def predict_links(distances: np.ndarray, n_links: int = 10, candidate_edges: list = None) -> list:
    """
    Predict links based on hyperbolic distances.

    Args:
        distances: Distance matrix between all nodes
        n_links: Number of links to predict
        candidate_edges: List of candidate edge pairs to consider for prediction (e.g., Ω_R + Ω_N)
                        If provided, only these pairs will be considered for prediction

    Returns:
        List of predicted links as (u, v, distance) tuples, sorted by distance (smallest first)
    """

    if candidate_edges is not None:
        # Use specific candidate edges (e.g., Ω_R + Ω_N)
        candidate_distances = []

        for u, v in candidate_edges:
            dist = distances[u, v]
            candidate_distances.append((u, v, dist))

        # Sort by distance and take the n_links with smallest distances
        candidate_distances.sort(key=lambda x: x[2])
        predicted_links = candidate_distances[:n_links]

    else:
        # Original behavior: consider all possible pairs
        # Create a copy of the distance matrix
        dist_matrix = distances.copy()

        # Set diagonal to infinity to exclude self-loops
        np.fill_diagonal(dist_matrix, np.inf)

        # Only consider upper triangle to avoid duplicates (i < j)
        upper_triangle_indices = np.triu_indices_from(dist_matrix, k=1)
        upper_distances = dist_matrix[upper_triangle_indices]

        # Find the N smallest distances in upper triangle
        sorted_indices = np.argsort(upper_distances)[:n_links]
        row_indices = upper_triangle_indices[0][sorted_indices]
        col_indices = upper_triangle_indices[1][sorted_indices]
        predicted_distances = upper_distances[sorted_indices]

        # Create list of predicted links
        predicted_links = [(int(row), int(col), float(dist)) for row, col, dist in zip(row_indices, col_indices, predicted_distances)]

    return predicted_links


def evaluate_predictions(predicted_links: list, omega_R: list, omega_N: list) -> dict:
    """Evaluate link prediction performance."""

    # Extract predicted node pairs
    predicted_pairs = [(u, v) for u, v, _ in predicted_links]

    # Calculate metrics

    true_positives = sum(1 for pair in predicted_pairs if pair in omega_R)
    false_positives = sum(1 for pair in predicted_pairs if pair in omega_N)
    false_negatives = sum(1 for pair in omega_R if pair not in predicted_pairs)
    true_negatives = sum(1 for pair in omega_N if pair not in predicted_pairs)

    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }

    return metrics


def plot_embeddings(
    poincare_embeddings: np.ndarray,
    embedding_runner: HyperbolicEmbeddings,
    edges: list,
    true_positives: list,
    false_positives: list,
    plot_title: str,
    save_path: str,
):

    # Embeddings with predictions (in Poincaré disk)
    x, y = poincare_embeddings[:, 0], poincare_embeddings[:, 1]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(plot_title)

    # Plot omega_E edges
    for u, v in edges:
        if u < len(x) and v < len(x):
            p1 = (x[u], y[u])
            p2 = (x[v], y[v])

            embedding_runner.plot_geodesic_arc(p1, p2, ax)

    # Plot recovered links
    for u, v, _ in true_positives:
        if u < len(x) and v < len(x):
            p1 = (x[u], y[u])
            p2 = (x[v], y[v])

            embedding_runner.plot_geodesic_arc(p1, p2, ax, color="green", linestyle="solid", linewidth=2)

    # Plot false positives
    for u, v, _ in false_positives:
        if u < len(x) and v < len(x):
            p1 = (x[u], y[u])
            p2 = (x[v], y[v])

            embedding_runner.plot_geodesic_arc(p1, p2, ax, color="red", linestyle="solid", linewidth=2)

    ax.scatter(x, y, s=150, edgecolor="black", color="skyblue", zorder=2)

    for i in range(len(x)):
        ax.text(x[i], y[i], str(i), fontsize=10, ha="center", va="center", zorder=3)

    # Draw Poincaré disk boundary
    ax.add_artist(plt.Circle((0, 0), 1, fill=False, color="black", linestyle="--"))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Link prediction pipeline example")
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="poincare_maps",
        choices=["poincare_embeddings", "lorentz", "poincare_maps", "dmercator", "hydra", "hypermap", "hydra_plus"],
        help="Type of hyperbolic embedding to use",
    )
    parser.add_argument("--n_links", type=int, default=None, help="Number of links to predict (default: same as number of edges in graph)")
    parser.add_argument("--q", type=float, default=0.5, help="Probability of keeping a link (default: 0.5)")

    return parser.parse_args()


def main():
    """Run a simple example of the link prediction pipeline."""
    args = parse_args()
    PATH = f"test/tree_test/plots/link_prediction/"

    print("Link Prediction Pipeline Example")
    print("=" * 40)
    print(f"Embedding type: {args.embedding_type}")
    print()

    # Load CORA graph
    with open("./data/Cora/cora_graph.pkl", "rb") as f:
        edge_list = pickle.load(f)
    with open("./data/Cora/cora_graph.json", "r") as f:
        graph_data = json.load(f)

    # Build networkx graph from edge index
    graph = nx.Graph()
    graph.add_edges_from(edge_list)

    A = nx.to_numpy_array(graph)

    labels = graph_data["y"]

    # Train hyperbolic embeddings
    print("Training hyperbolic embeddings for original graph...")
    configurations = {
        "poincare_embeddings": {"dim": 2, "negs": 5, "epochs": 1000, "batch_size": 256, "dimension": 1},
        "lorentz": {"dim": 2, "epochs": 10000, "batch_size": 1024, "num_nodes": len(graph.nodes)},
        "dmercator": {"dim": 1},
        "hydra": {"dim": 2},
        "poincare_maps": {"dim": 2, "epochs": 1000},
        "hypermap": {"dim": 3},
        "hydra_plus": {"dim": 2},
    }
    config = configurations[args.embedding_type]
    embedding_runner = HyperbolicEmbeddings(embedding_type=args.embedding_type, config=config)

    # Prepare training data
    adjacency_matrix = nx.to_numpy_array(graph)

    # Train the model
    model_path = f"saved_models/tree_test/link_prediction/{args.embedding_type}_original_graph_model.bin"
    embedding_runner.train(adjacency_matrix=adjacency_matrix, model_path=model_path)

    # Get embeddings
    original_graph_embeddings = embedding_runner.get_all_embeddings(model_path)
    print(f"Embeddings shape: {original_graph_embeddings.shape}")

    native_space = embedding_runner.model.native_space

    # Convert embeddings to Poincaré for better visualization
    if native_space != "poincare":
        original_graph_poincare_embeddings = convert_coordinates(original_graph_embeddings, native_space, "poincare")
    else:
        original_graph_poincare_embeddings = original_graph_embeddings

    plot_title = "Original Graph Embeddings in Poincaré Disk"
    save_path = f"{PATH}/{args.embedding_type}_original_graph_embeddings.pdf"
    plot_embeddings(original_graph_poincare_embeddings, embedding_runner, graph.edges(), [], [], plot_title, save_path)

    # Generating graph from Distance Matrix
    print("Generating graph from distance matrix...")

    # Convert to hyperboloid coordinates if needed
    if native_space != "hyperboloid":
        print(f"Converting embeddings from {native_space} to hyperboloid coordinates")
        orginal_graph_hyperboloid_embeddings = convert_coordinates(original_graph_embeddings, native_space, "hyperboloid")
    else:
        orginal_graph_hyperboloid_embeddings = original_graph_embeddings

    orginal_graph_distance_matrix = compute_distances(orginal_graph_hyperboloid_embeddings, space="hyperboloid")

    distance_matrix_predicted_links = predict_links(orginal_graph_distance_matrix, n_links=len(graph.edges()))

    print(f"Distance matrix predicted links: {distance_matrix_predicted_links}")

    # Analyze predictions: recovered links vs false positives
    recovered_links = []  # Correctly predicted removed links (from Ω_R)
    false_positives = []  # Incorrectly predicted non-links (from Ω_N)
    true_edges_set = set(graph.edges())
    true_edges_set.update([(v, u) for u, v in graph.edges()])  # Add both directions

    for u, v, dist in distance_matrix_predicted_links:
        if (u, v) in true_edges_set:
            recovered_links.append((u, v, dist))
        else:
            false_positives.append((u, v, dist))

    print("\nCreating visualization...")

    # Convert embeddings to Poincaré for better visualization
    if native_space != "poincare":
        original_graph_poincare_embeddings = convert_coordinates(orginal_graph_hyperboloid_embeddings, "hyperboloid", "poincare")
    else:
        original_graph_poincare_embeddings = original_graph_embeddings

    plot_title = "Distance Matrix Predicted Links in Poincaré Disk"
    save_path = f"{PATH}/{args.embedding_type}_distance_matrix_predicted_links.pdf"
    plot_embeddings(
        original_graph_poincare_embeddings, embedding_runner, graph.edges(), recovered_links, false_positives, plot_title, save_path
    )

    ########################################################################################################################################################

    # Simulate link removal
    print("Simulating link removal...")
    results = simulate_link_removal(graph, q=args.q)
    omega_E = results["omega_E"]  # List of remaining links

    # Create graph with same nodes as original graph and edges from omega_E
    graph_from_omega_E = nx.Graph()
    graph_from_omega_E.add_nodes_from(graph.nodes())  # Preserve all original nodes
    graph_from_omega_E.add_edges_from(omega_E)  # Add only remaining edges

    print(f"Graph created from omega_E: {len(graph_from_omega_E.nodes)} nodes, {len(graph_from_omega_E.edges)} edges")

    # Train hyperbolic embeddings
    print("Training hyperbolic embeddings...")
    configurations = {
        "poincare_embeddings": {"dim": 2, "negs": 5, "epochs": 1000, "batch_size": 256, "dimension": 1},
        "lorentz": {"dim": 2, "epochs": 10000, "batch_size": 1024, "num_nodes": len(graph.nodes)},
        "dmercator": {"dim": 1},
        "hydra": {"dim": 2},
        "poincare_maps": {"dim": 2, "epochs": 1000},
        "hypermap": {"dim": 3},
        "hydra_plus": {"dim": 2},
    }
    config = configurations[args.embedding_type]
    embedding_runner = HyperbolicEmbeddings(embedding_type=args.embedding_type, config=config)

    # Prepare training data
    adjacency_matrix = nx.to_numpy_array(graph_from_omega_E)

    # Train the model
    model_path = f"saved_models/tree_test/link_prediction/{args.embedding_type}_model.bin"
    embedding_runner.train(adjacency_matrix=adjacency_matrix, model_path=model_path)

    # Get embeddings
    embeddings = embedding_runner.get_all_embeddings(model_path)
    print(f"Embeddings shape: {embeddings.shape}")

    # Convert embeddings to Poincaré for better visualization
    if native_space != "poincare":
        poincare_embeddings = convert_coordinates(embeddings, native_space, "poincare")
    else:
        poincare_embeddings = embeddings

    plot_title = "Distance Matrix Predicted Links in Poincaré Disk"
    save_path = f"{PATH}/{args.embedding_type}_omega_E_graph_embeddings.pdf"
    plot_embeddings(poincare_embeddings, embedding_runner, omega_E, [], [], plot_title, save_path)

    # Convert to hyperboloid coordinates if needed
    native_space = embedding_runner.model.native_space
    if native_space != "hyperboloid":
        print(f"Converting embeddings from {native_space} to hyperboloid coordinates")
        hyperboloid_embeddings = convert_coordinates(embeddings, native_space, "hyperboloid")
    else:
        hyperboloid_embeddings = embeddings

    # Compute hyperbolic distances
    print("Computing hyperbolic distances...")
    distances = compute_distances(hyperboloid_embeddings, space="hyperboloid")
    print(f"Distance matrix shape: {distances.shape}")
    print(f"Distance range: {distances[distances > 0].min():.4f} to {distances.max():.4f}")

    # Predict links from Ω_R + Ω_N (removed links + true non-links)
    omega_R = results["omega_R"]
    omega_N = results["omega_N"]
    omega_R_plus_N = results["omega_R_plus_N"]
    n_links = args.n_links if args.n_links else len(omega_R)
    print(f"Number of links to predict: {n_links}")

    predicted_links = predict_links(distances, n_links=n_links, candidate_edges=omega_R_plus_N)

    # Evaluate predictions against removed edges (Ω_R) - the true positive targets
    print()
    print("Evaluating predictions...")
    metrics = evaluate_predictions(predicted_links, omega_R, omega_N)

    # Print results
    print("\nLink Prediction Results:")
    print(f"Predicted {len(predicted_links)} links from {len(omega_R_plus_N)} candidates (Ω_R + Ω_N)")
    print(f"Target removed edges (Ω_R): {len(omega_R)} links")
    print(f"True non-links (Ω_N): {len(results['omega_N'])} pairs")
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Analyze predictions: recovered links vs false positives
    recovered_links = []  # Correctly predicted removed links (from Ω_R)
    false_positives = []  # Incorrectly predicted non-links (from Ω_N)
    omega_R_set = set(omega_R)
    omega_R_set.update([(v, u) for u, v in omega_R])  # Add both directions

    for u, v, dist in predicted_links:
        if (u, v) in omega_R_set:
            recovered_links.append((u, v, dist))
        else:
            false_positives.append((u, v, dist))

    print(f"\nPrediction Analysis:")

    print("\nRecovered links (node1, node2, distance):")
    for u, v, dist in recovered_links:
        print(f"  ({u}, {v}): {dist:.4f}")

    print("\nFalse positives (node1, node2, distance):")
    for u, v, dist in false_positives:
        print(f"  ({u}, {v}): {dist:.4f}")

    true_links = []
    for u, v in omega_R:
        dist = distances[u, v]
        true_links.append((u, v, dist))

    print("\nTrue links (Omega R) (node1, node2, distance):")
    for u, v, dist in true_links:
        print(f"  ({u}, {v}): {dist:.4f}")

    # Create a visualization showing predicted links
    print("\nCreating visualization...")

    # Convert embeddings to Poincaré for better visualization
    if native_space != "poincare":
        poincare_embeddings = convert_coordinates(hyperboloid_embeddings, "hyperboloid", "poincare")
    else:
        poincare_embeddings = embeddings

    plot_title = "Predicted Links in Poincaré Disk"
    save_path = f"{PATH}/{args.embedding_type}_omege_E_predicted_links.pdf"
    plot_embeddings(poincare_embeddings, embedding_runner, omega_E, recovered_links, false_positives, plot_title, save_path)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
