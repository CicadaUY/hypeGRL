import argparse
import multiprocessing
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.gridspec import GridSpec

from hyperbolic_embeddings import HyperbolicEmbeddings

multiprocessing.set_start_method("spawn", force=True)


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

    ax.set_title(title, fontsize=30)

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
        if labels:
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
                    label=label,
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

        # Reduce radial grid lines (only 3 concentric circles)
        ax.set_rticks([boundary_radius * 0.33, boundary_radius * 0.66, boundary_radius])

        # Remove angle labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

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
        if labels:
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
                    label=label,
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


def get_model_path(model_dir, embedding_type, run_id):
    """
    Get the model path for a given embedding type and run ID.
    Handles special case for dmercator which uses .inf_coord extension.
    Note: poincare_embeddings naturally results in double "embeddings" in filename
    (embedding_type="poincare_embeddings" -> "poincare_embeddings_embeddings_run0.bin")
    """
    base_path = os.path.join(model_dir, f"{embedding_type}_embeddings_run{run_id}.bin")

    # For dmercator, check if .inf_coord file exists
    if embedding_type == "dmercator":
        coord_path = base_path + ".inf_coord"
        if os.path.exists(coord_path):
            return base_path  # Return base path, .inf_coord is handled internally
        else:
            return None

    # For other models, check if the base file exists
    if os.path.exists(base_path):
        return base_path
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="Load and compare pre-trained embedding models with subplots.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="saved_models/tree_test",
        help="Directory containing the trained models.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="test/tree_test/plots/embeddings",
        help="Directory to save the plots.",
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
        help="Space to plot embeddings in.",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=0,
        help="Run ID to load (default: 0).",
    )

    args = parser.parse_args()

    # Build tree graph (needed for edge_list in plotting)
    G = nx.balanced_tree(2, 4)
    edge_list = list(G.edges())
    labels = list(range(G.number_of_nodes()))

    # Embedding configs (needed to initialize the models)
    configurations = {
        "poincare_embeddings": {
            "dim": 2,
            "negs": 5,
            "epochs": 1000,
            "batch_size": 256,
            "dimension": 1,
        },
        "lorentz": {"dim": 2, "epochs": 10000, "batch_size": 1024, "num_nodes": 31},
        "dmercator": {"dim": 1},
        "hydra": {"dim": 2},
        "poincare_maps": {"dim": 2, "epochs": 1000},
        "hypermap": {"dim": 3},
        "hydra_plus": {"dim": 2},
    }

    os.makedirs(args.plot_dir, exist_ok=True)

    # Generate comparison plot with all embedding types
    embedding_types = [
        "hydra",
        "hydra_plus",
        "hypermap",
        "dmercator",
        "poincare_embeddings",
        "poincare_maps",
        "lorentz",
    ]

    # Display names with timing information
    embedding_display_names = {
        "hydra": "Hydra\n(0.003 s ±0.001)",
        "hydra_plus": "Hydra+\n(0.02 s ±0.01)",
        "hypermap": "HyperMap\n(0.04 s ±0.02)",
        "dmercator": "D-Mercator\n(0.08 s ±0.01)",
        "poincare_embeddings": "Poincaré Embeddings\n(1.08 s ±0.06)",
        "poincare_maps": "Poincaré Maps\n(3.27 s ±0.15)",
        "lorentz": "Lorentz Embeddings\n(18367.3 s ±1.2)",
    }

    # Create figure with subplots (2 rows, 4 in first row, 3 centered in second row)
    # Use GridSpec for flexible layout with centered second row
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(2, 8, figure=fig)
    axes = []

    if args.output_space.lower() == "spherical":
        # First row: 4 plots, each spanning 2 columns
        for i in range(4):
            axes.append(fig.add_subplot(gs[0, i * 2 : (i * 2) + 2], projection="polar"))
        # Second row: 3 plots, each spanning 2 columns, offset by 1 to center
        for i in range(3):
            axes.append(fig.add_subplot(gs[1, (i * 2) + 1 : (i * 2) + 3], projection="polar"))
    else:
        # First row: 4 plots, each spanning 2 columns
        for i in range(4):
            axes.append(fig.add_subplot(gs[0, i * 2 : (i * 2) + 2]))
        # Second row: 3 plots, each spanning 2 columns, offset by 1 to center
        for i in range(3):
            axes.append(fig.add_subplot(gs[1, (i * 2) + 1 : (i * 2) + 3]))

    all_embeddings_data = []
    failed_models = []

    # Load all embeddings and collect data
    for idx, embedding_type in enumerate(embedding_types):
        print(f"\n{'='*60}")
        print(f"Loading {embedding_type} embeddings ({idx+1}/{len(embedding_types)})...")
        print(f"{'='*60}")

        config = configurations[embedding_type]
        model_path = get_model_path(args.model_dir, embedding_type, args.run_id)

        # Check if model file exists
        if model_path is None:
            print(f"  ✗ Model file not found for {embedding_type}")
            failed_models.append(embedding_type)
            continue

        # For dmercator, also check that .inf_coord exists
        if embedding_type == "dmercator":
            coord_path = model_path + ".inf_coord"
            if not os.path.exists(coord_path):
                print(f"  ✗ Model file not found: {coord_path}")
                failed_models.append(embedding_type)
                continue

        try:
            # Initialize embedding runner (no training needed)
            embedding_runner = HyperbolicEmbeddings(embedding_type=embedding_type, config=config)

            # Load embeddings from disk
            embeddings = embedding_runner.get_all_embeddings(model_path)
            native_embedding_space = embedding_runner.model.native_space

            print(f"  ✓ Loaded {embedding_type} embeddings (native space: {native_embedding_space})")
            print(f"    Embeddings shape: {embeddings.shape}")

            all_embeddings_data.append(
                {
                    "type": embedding_type,
                    "embeddings": embeddings,
                    "native_space": native_embedding_space,
                    "runner": embedding_runner,
                }
            )

        except Exception as e:
            print(f"  ✗ Error loading {embedding_type}: {e}")
            failed_models.append(embedding_type)
            continue

    # Report any failed models
    if failed_models:
        print(f"\n{'='*60}")
        print(f"WARNING: Failed to load {len(failed_models)} model(s):")
        for model in failed_models:
            print(f"  - {model}")
        print(f"{'='*60}")

    if not all_embeddings_data:
        print("\n✗ No models were successfully loaded. Exiting.")
        return

    # Plot all embeddings in subplots
    print(f"\n{'='*60}")
    print(f"Generating comparison plot in {args.output_space} coordinates...")
    print(f"{'='*60}")

    for idx, data in enumerate(all_embeddings_data):
        ax_idx = axes[idx]
        plot_embeddings_on_axis(
            ax=ax_idx,
            embedding_runner=data["runner"],
            embeddings=data["embeddings"],
            native_embedding_space=data["native_space"],
            output_space=args.output_space,
            labels=labels,
            edge_list=edge_list,
            title=embedding_display_names.get(data["type"], data["type"].replace("_", " ").title()),
            point_size=80,
            show_node_labels=False,
            node_label_size=6,
            edge_alpha=0.4,
            edge_width=0.8,
            colormap="viridis",
        )

    # No unused subplots to hide since we created exactly 7 with GridSpec
    plt.tight_layout()

    plot_path = os.path.join(args.plot_dir, f"all_embeddings_comparison_{args.output_space}.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Comparison plot saved to {plot_path}")
    plt.close()

    print(f"\n{'='*60}")
    print(f"Successfully loaded and plotted {len(all_embeddings_data)} embedding type(s)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
