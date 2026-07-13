import argparse
import multiprocessing
import os
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from gaussian_loader import load_gaussian_dataset
from hyperbolic_embeddings import HyperbolicEmbeddings

multiprocessing.set_start_method("spawn", force=True)


def load_polblogs_dataset():
    """
    Load PolBlogs dataset from PyTorch Geometric and convert to NetworkX format.

    Returns:
    --------
    graph : nx.Graph
        NetworkX graph representation
    labels : np.ndarray
        Node labels (numeric)
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

    print(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Number of classes: {len(np.unique(labels))}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Extract largest connected component (needed for dmercator)
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

        # Filter labels to match the largest component
        labels = labels[sorted(largest_cc)]

        print(
            f"Largest component: {graph.number_of_nodes()} nodes ({100*len(largest_cc)/data.num_nodes:.1f}% of original), {graph.number_of_edges()} edges"
        )
        print(f"Number of classes: {len(np.unique(labels))}")
        print(f"Label distribution: {np.bincount(labels)}")

    return graph, labels


def load_gaussian_hyperbolic_dataset(
    num_samples: int = 1250,
    num_classes: int = 2,
    dimension: int = 2,
    k_neighbors: int = 10,
    seed: int = 42,
    noise_std: float = 2.0,
    separation: float = 2.0,
):
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

    return graph, labels


def plot_embeddings_on_axis(
    ax,
    embedding_runner,
    embeddings,
    native_embedding_space,
    output_space,
    labels,
    edge_list,
    title,
    point_size=100,
    show_node_labels=False,
    node_label_size=8,
    edge_alpha=0.6,
    edge_width=1.0,
    colormap="viridis",
):
    """Plot embeddings on a given matplotlib axis (for subplots)."""
    # Convert coordinates if needed
    if native_embedding_space != output_space:
        plot_embeddings = embedding_runner.convert_coordinates(embeddings, native_embedding_space, output_space)
    else:
        plot_embeddings = embeddings

    # Validate embeddings
    embedding_runner.validate_embeddings(plot_embeddings, output_space)

    x, y = plot_embeddings[:, 0], plot_embeddings[:, 1]

    ax.set_title(title, fontsize=12, fontweight="bold")

    if output_space.lower() == "spherical":
        # Calculate boundary radius first for the background circle
        max_radius = np.max(x) if len(x) > 0 else 1.0
        boundary_radius = max_radius * 1.1

        # Plot edges
        if edge_list:
            for u, v in edge_list:
                if u < len(x) and v < len(x):
                    p1 = (y[u], x[u])
                    p2 = (y[v], x[v])
                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        color="gray",
                        linewidth=edge_width,
                        alpha=edge_alpha,
                        zorder=1,
                    )

        # Plot points
        if labels is not None:
            labels_arr = np.array(labels)
            unique_labels = sorted(set(labels_arr))
            cmap_obj = cm.get_cmap(colormap, len(unique_labels))
            label_to_color = {label: cmap_obj(i) for i, label in enumerate(unique_labels)}

            for label in unique_labels:
                indices = np.where(labels_arr == label)[0]
                ax.scatter(
                    y[indices],
                    x[indices],
                    s=point_size,
                    edgecolor="black",
                    color=label_to_color[label],
                    label=f"Class {label}",
                    zorder=2,
                )
        else:
            ax.scatter(y, x, s=point_size, edgecolor="black", color="skyblue", zorder=2)

        # Add node labels
        if show_node_labels:
            for i in range(len(x)):
                ax.text(
                    y[i],
                    x[i],
                    str(i),
                    fontsize=node_label_size,
                    ha="center",
                    va="center",
                    zorder=3,
                )

        # Set limits (boundary_radius already calculated above)
        ax.set_ylim(0, boundary_radius)
        ax.set_rlim(0, boundary_radius)

    else:  # poincare
        # Plot edges
        if edge_list:
            for u, v in edge_list:
                if u < len(x) and v < len(x):
                    p1 = (x[u], y[u])
                    p2 = (x[v], y[v])
                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        color="gray",
                        linewidth=edge_width,
                        alpha=edge_alpha,
                        zorder=1,
                    )

        # Plot points
        if labels is not None:
            labels_arr = np.array(labels)
            unique_labels = sorted(set(labels_arr))
            cmap_obj = cm.get_cmap(colormap, len(unique_labels))
            label_to_color = {label: cmap_obj(i) for i, label in enumerate(unique_labels)}

            for label in unique_labels:
                indices = np.where(labels_arr == label)[0]
                ax.scatter(
                    x[indices],
                    y[indices],
                    s=point_size,
                    edgecolor="black",
                    color=label_to_color[label],
                    label=f"Class {label}",
                    zorder=2,
                )
        else:
            ax.scatter(x, y, s=point_size, edgecolor="black", color="skyblue", zorder=2)

        # Add node labels
        if show_node_labels:
            for i in range(len(x)):
                ax.text(
                    x[i],
                    y[i],
                    str(i),
                    fontsize=node_label_size,
                    ha="center",
                    va="center",
                    zorder=3,
                )

        # Draw grey background circle (Poincaré disk boundary)
        ax.add_artist(plt.Circle((0, 0), 1, fill=False, color="gray", linewidth=1.5, zorder=0))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Compare dmercator vs poincare_maps embeddings for gaussian and polblogs datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gaussian",
        choices=["gaussian", "polblogs"],
        help="Dataset to use (gaussian or polblogs).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory to save the trained models (default: saved_models/{dataset}_comparison).",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="test/other/plots/embeddings",
        help="Directory to save the plots.",
    )
    parser.add_argument(
        "--output_space",
        type=str,
        default="poincare",
        choices=[
            "poincare",
            "hyperboloid",
            "klein",
            "hemisphere",
            "half_plane",
            "spherical",
        ],
        help="Space to plot embeddings in.",
    )
    parser.add_argument(
        "--gaussian_num_samples",
        type=int,
        default=1250,
        help="Number of samples for Gaussian dataset.",
    )
    parser.add_argument(
        "--gaussian_num_classes",
        type=int,
        default=6,
        help="Number of classes for Gaussian dataset.",
    )
    parser.add_argument(
        "--gaussian_dimension",
        type=int,
        default=2,
        help="Dimension for Gaussian dataset.",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=10,
        help="Number of neighbors for k-NN graph construction (for Gaussian dataset).",
    )
    parser.add_argument(
        "--gaussian_seed",
        type=int,
        default=42,
        help="Random seed for Gaussian dataset generation.",
    )
    parser.add_argument(
        "--show_edges",
        action="store_true",
        help="Show edges in the plots.",
    )
    parser.add_argument(
        "--show_legend",
        action="store_true",
        help="Show legend in the plots.",
    )

    args = parser.parse_args()

    # Set default model directory if not provided
    if args.model_dir is None:
        args.model_dir = f"saved_models/{args.dataset}_comparison"

    # Load dataset
    print(f"\n{'='*80}")
    print(f"Loading {args.dataset} dataset...")
    print(f"{'='*80}")

    if args.dataset == "gaussian":
        G, labels = load_gaussian_hyperbolic_dataset(
            num_samples=args.gaussian_num_samples,
            num_classes=args.gaussian_num_classes,
            dimension=args.gaussian_dimension,
            k_neighbors=args.k_neighbors,
            seed=args.gaussian_seed,
        )
    elif args.dataset == "polblogs":
        G, labels = load_polblogs_dataset()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Get edge list for training (always use full edge list for training)
    edge_list = list(G.edges())

    # Get edge list for plotting (can be empty if --show_edges is not passed)
    plot_edge_list = edge_list if args.show_edges else []

    # Embedding configurations
    configurations = {
        "dmercator": {"dim": 2},
        "poincare_maps": {"dim": 2, "epochs": 1000},
    }

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    # Generate comparison plot with both embedding types
    embedding_types = ["dmercator", "poincare_maps"]

    # Create figure with subplots (1 row x 2 columns)
    # Use polar projection for spherical coordinates
    if args.output_space.lower() == "spherical":
        fig = plt.figure(figsize=(14, 6))
        axes = []
        for i in range(2):
            axes.append(fig.add_subplot(1, 2, i + 1, projection="polar"))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    all_embeddings_data = []
    timing_results = {}  # Will store time for each embedding type

    # Prepare adjacency matrix (needed for poincare_maps)
    A = nx.to_numpy_array(G)

    # Train all embeddings and collect data
    for idx, embedding_type in enumerate(embedding_types):
        print(f"\n{'='*80}")
        print(f"Training {embedding_type} embeddings ({idx+1}/{len(embedding_types)})...")
        print(f"{'='*80}")

        config = configurations[embedding_type]
        model_path = os.path.join(args.model_dir, f"{embedding_type}_embeddings.bin")

        embedding_runner = HyperbolicEmbeddings(embedding_type=embedding_type, config=config)

        # Start timing
        start_time = time.time()

        if embedding_type == "poincare_maps":
            embedding_runner.train(adjacency_matrix=A, model_path=model_path)
        else:
            embedding_runner.train(edge_list=edge_list, model_path=model_path)

        embeddings = embedding_runner.get_all_embeddings(model_path)

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        timing_results[embedding_type] = elapsed_time

        print(f"\n✓ {embedding_type} completed: {elapsed_time:.2f}s")

        native_embedding_space = embedding_runner.model.native_space
        all_embeddings_data.append(
            {
                "type": embedding_type,
                "embeddings": embeddings,
                "native_space": native_embedding_space,
                "runner": embedding_runner,
            }
        )

    # Plot all embeddings in subplots
    print(f"\n{'='*80}")
    print(f"Generating comparison plot in {args.output_space} coordinates...")
    print(f"{'='*80}")

    for idx, data in enumerate(all_embeddings_data):
        ax = axes[idx]
        plot_embeddings_on_axis(
            ax=ax,
            embedding_runner=data["runner"],
            embeddings=data["embeddings"],
            native_embedding_space=data["native_space"],
            output_space=args.output_space,
            labels=labels,
            edge_list=plot_edge_list,
            title=data["type"].replace("_", " ").title(),
            point_size=50 if args.dataset == "polblogs" else 80,
            show_node_labels=False,
            node_label_size=6,
            edge_alpha=0.2,
            edge_width=0.5,
            colormap="tab10",
        )

        # Add legend if requested
        if args.show_legend:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    plot_path = os.path.join(args.plot_dir, f"{args.dataset}_comparison_{args.output_space}.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Comparison plot saved to {plot_path}")
    plt.close()

    # Print timing summary
    print(f"\n{'='*80}")
    print(f"TIMING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Embedding Type':<25} {'Time (s)':<15}")
    print(f"{'-'*80}")

    # Sort by time
    sorted_timings = sorted(timing_results.items(), key=lambda x: x[1])

    for embedding_type, elapsed_time in sorted_timings:
        print(f"{embedding_type:<25} {elapsed_time:>10.2f}")

    print(f"{'-'*80}")
    total_time = sum(timing_results.values())
    print(f"{'TOTAL':<25} {total_time:>10.2f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
