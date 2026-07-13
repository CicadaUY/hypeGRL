"""
Train hyperbolic embeddings on various datasets and plot in spherical coordinates.

This script:
1. Loads a dataset (Gaussian or PolBlogs)
2. Trains hyperbolic embeddings (Poincare Maps or dmercator) on the dataset
3. Plots the embeddings in spherical coordinates
"""

import argparse
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from gaussian_loader import load_gaussian_dataset
from hyperbolic_embeddings import HyperbolicEmbeddings


def load_polblogs_dataset():
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
    show_node_labels=True,
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

    ax.set_title(title, fontsize=12)

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

        # Add legend
        if labels is not None:
            ax.legend(loc="upper right", fontsize=8)

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

        # Add legend
        if labels is not None:
            ax.legend(loc="upper right", fontsize=8)


def main():
    parser = argparse.ArgumentParser(description="Train hyperbolic embeddings on various datasets and plot in spherical coordinates")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gaussian",
        choices=["gaussian", "polblogs"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="poincare_maps",
        choices=["poincare_maps", "dmercator"],
        help="Type of embedding to train",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1250,
        help="Number of samples in the Gaussian dataset (only used for gaussian)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=6,
        help="Number of classes in the Gaussian dataset (only used for gaussian)",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=2,
        help="Dimension of the hyperbolic space (only used for gaussian)",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=10,
        help="Number of neighbors for k-NN graph construction (only used for gaussian)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset generation (only used for gaussian)",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=2.0,
        help="Standard deviation of cluster noise (only used for gaussian)",
    )
    parser.add_argument(
        "--separation",
        type=float,
        default=2.0,
        help="Separation between cluster centers (only used for gaussian)",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=2,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (only used for Poincare Maps)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory to save the trained model (default: saved_models/{dataset}_{embedding_type})",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Directory to save the plots (default: test/{dataset}_plots)",
    )
    parser.add_argument(
        "--output_space",
        type=str,
        default="spherical",
        choices=[
            "poincare",
            "hyperboloid",
            "klein",
            "hemisphere",
            "half_plane",
            "spherical",
        ],
        help="Space to plot embeddings in",
    )
    parser.add_argument(
        "--show_node_labels",
        action="store_true",
        help="Show node labels on the plot",
    )
    parser.add_argument(
        "--edge_alpha",
        type=float,
        default=0.3,
        help="Alpha value for edges",
    )
    parser.add_argument(
        "--edge_width",
        type=float,
        default=0.5,
        help="Width of edges",
    )
    parser.add_argument(
        "--point_size",
        type=int,
        default=50,
        help="Size of points in the plot",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="tab10",
        help="Colormap for class colors",
    )

    args = parser.parse_args()

    # Set default directories if not provided
    if args.model_dir is None:
        args.model_dir = f"saved_models/{args.dataset}_{args.embedding_type}"
    if args.plot_dir is None:
        args.plot_dir = f"test/{args.dataset}_plots"

    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    print("=" * 80)
    print(f"TRAINING {args.embedding_type.upper()} ON {args.dataset.upper()} DATASET")
    print("=" * 80)

    # Load dataset
    print("\n" + "-" * 80)
    print(f"STEP 1: Loading {args.dataset.upper()} Dataset")
    print("-" * 80)

    if args.dataset == "gaussian":
        print(f"\nDataset Configuration:")
        print(f"  Samples: {args.num_samples}")
        print(f"  Classes: {args.num_classes}")
        print(f"  Dimension: {args.dimension}")
        print(f"  k-NN neighbors: {args.k_neighbors}")
        print(f"  Noise std: {args.noise_std}")
        print(f"  Separation: {args.separation}")
        print(f"  Seed: {args.seed}")

        graph, labels, node_indices = load_gaussian_dataset(
            dimension=args.dimension,
            num_samples=args.num_samples,
            num_classes=args.num_classes,
            k_neighbors=args.k_neighbors,
            seed=args.seed,
            noise_std=args.noise_std,
            separation=args.separation,
        )
    elif args.dataset == "polblogs":
        graph, labels, node_indices = load_polblogs_dataset()

    edge_list = list(graph.edges())

    # Train embeddings
    print("\n" + "-" * 80)
    print(f"STEP 2: Training {args.embedding_type.upper()} Embeddings")
    print("-" * 80)
    print(f"\nEmbedding Configuration:")
    print(f"  Embedding type: {args.embedding_type}")
    print(f"  Embedding dimension: {args.embedding_dim}")
    if args.embedding_type == "poincare_maps":
        print(f"  Epochs: {args.epochs}")

    # Configure based on embedding type
    if args.embedding_type == "poincare_maps":
        config = {
            "dim": args.embedding_dim,
            "epochs": args.epochs,
        }
    elif args.embedding_type == "dmercator":
        config = {
            "dim": args.embedding_dim,
        }

    embedding_runner = HyperbolicEmbeddings(embedding_type=args.embedding_type, config=config)

    # Train the model
    model_path = os.path.join(args.model_dir, f"{args.embedding_type}_embeddings.bin")
    print(f"\nTraining model (saving to {model_path})...")

    if args.embedding_type == "poincare_maps":
        # Poincare Maps uses adjacency matrix
        A = nx.to_numpy_array(graph)
        embedding_runner.train(adjacency_matrix=A, model_path=model_path)
    elif args.embedding_type == "dmercator":
        # dmercator uses edge list
        embedding_runner.train(edge_list=edge_list, model_path=model_path)

    # Get embeddings
    embeddings = embedding_runner.get_all_embeddings(model_path)
    native_space = embedding_runner.model.native_space

    print(f"\n✓ Training completed!")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Native space: {native_space}")

    # Plot embeddings
    print("\n" + "-" * 80)
    print("STEP 3: Plotting Embeddings in Spherical Coordinates")
    print("-" * 80)
    print(f"\nPlot Configuration:")
    print(f"  Output space: {args.output_space}")
    print(f"  Point size: {args.point_size}")
    print(f"  Edge alpha: {args.edge_alpha}")
    print(f"  Edge width: {args.edge_width}")
    print(f"  Show node labels: {args.show_node_labels}")
    print(f"  Colormap: {args.colormap}")

    # Create figure
    if args.output_space.lower() == "spherical":
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="polar")
    else:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Plot embeddings
    embedding_name = args.embedding_type.replace("_", " ").title()
    plot_embeddings_on_axis(
        ax=ax,
        embedding_runner=embedding_runner,
        embeddings=embeddings,
        native_embedding_space=native_space,
        output_space=args.output_space,
        labels=labels,
        edge_list=edge_list,
        title=f"{embedding_name} Embeddings ({args.output_space.title()} Coordinates)",
        point_size=args.point_size,
        show_node_labels=args.show_node_labels,
        node_label_size=6,
        edge_alpha=args.edge_alpha,
        edge_width=args.edge_width,
        colormap=args.colormap,
    )

    plt.tight_layout()

    # Save plot
    plot_filename = f"{args.dataset}_{args.embedding_type}_{args.output_space}.pdf"
    plot_path = os.path.join(args.plot_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Plot saved to: {plot_path}")

    # Also save as PNG
    plot_path_png = plot_path.replace(".pdf", ".png")
    plt.savefig(plot_path_png, dpi=300, bbox_inches="tight")
    print(f"✓ Plot also saved to: {plot_path_png}")

    plt.close()

    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nModel saved to: {model_path}")
    print(f"Plots saved to: {args.plot_dir}")


if __name__ == "__main__":
    main()
