import argparse
import multiprocessing
import os
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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
    plt.rcParams["lines.linewidth"] = 3
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 60
    plt.rcParams["lines.markersize"] = 25
    plt.rcParams["axes.grid"] = True

    ax.set_title(title, fontsize=20)

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


def main():
    parser = argparse.ArgumentParser(description="Hybrid approach: Load lorentz model, train all other embedding types with subplots.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="saved_models/tree_test",
        help="Directory to save/load the trained models.",
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
        "--num_runs",
        type=int,
        default=3,
        help="Number of runs for each embedding type to calculate mean and std of timing.",
    )
    parser.add_argument(
        "--lorentz_run_id",
        type=int,
        default=0,
        help="Run ID to load for lorentz model (default: 0).",
    )

    args = parser.parse_args()

    # Build tree graph
    G = nx.balanced_tree(2, 4)

    lengths = dict(nx.all_pairs_shortest_path_length(G))
    n = G.number_of_nodes()
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = lengths[i][j]
    A = nx.to_numpy_array(G)

    edge_list = list(G.edges())
    labels = list(range(n))

    # Embedding configs
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

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    # Generate comparison plot with all embedding types
    embedding_types = [
        "poincare_embeddings",
        "dmercator",
        "hydra",
        "hypermap",
        "hydra_plus",
        "poincare_maps",
        "lorentz",
    ]

    # Create figure with subplots (3 rows x 3 columns, using 7 subplots)
    # Use polar projection for spherical coordinates
    if args.output_space.lower() == "spherical":
        fig = plt.figure(figsize=(18, 18))
        axes = []
        for i in range(9):
            axes.append(fig.add_subplot(3, 3, i + 1, projection="polar"))
    else:
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()

    all_embeddings_data = []
    timing_results = {}  # Will store list of times for each embedding type (excluding lorentz)

    # Process all embeddings: load lorentz, train others
    for idx, embedding_type in enumerate(embedding_types):
        print(f"\n{'='*60}")

        config = configurations[embedding_type]

        # Special handling for lorentz: load from disk
        if embedding_type == "lorentz":
            print(f"Loading {embedding_type} embeddings from disk ({idx+1}/{len(embedding_types)})...")
            print(f"{'='*60}")

            lorentz_model_path = os.path.join(args.model_dir, f"lorentz_embeddings_run{args.lorentz_run_id}.bin")

            # Check if model file exists
            if not os.path.exists(lorentz_model_path):
                print(f"  ✗ Model file not found: {lorentz_model_path}")
                print(f"  ⚠ Skipping lorentz model. Continuing with other models...")
                continue

            try:
                # Initialize embedding runner (no training needed)
                embedding_runner = HyperbolicEmbeddings(embedding_type=embedding_type, config=config)

                # Load embeddings from disk
                embeddings = embedding_runner.get_all_embeddings(lorentz_model_path)
                native_embedding_space = embedding_runner.model.native_space

                print(f"  ✓ Loaded {embedding_type} embeddings (native space: {native_embedding_space})")
                print(f"    Embeddings shape: {embeddings.shape}")
                print(f"    Model loaded from: {lorentz_model_path}")

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
                print(f"  ⚠ Skipping lorentz model. Continuing with other models...")
                continue

        else:
            # For all other models: train them
            print(f"Training {embedding_type} embeddings ({idx+1}/{len(embedding_types)})...")
            print(f"Running {args.num_runs} times for timing statistics...")
            print(f"{'='*60}")

            timing_results[embedding_type] = []
            native_embedding_space = None

            # Run multiple times for timing statistics
            for run in range(args.num_runs):
                print(f"\n  Run {run+1}/{args.num_runs}...")
                model_path = os.path.join(args.model_dir, f"{embedding_type}_embeddings_run{run}.bin")

                embedding_runner = HyperbolicEmbeddings(embedding_type=embedding_type, config=config)

                # Start timing
                start_time = time.time()

                if embedding_type in ["hydra", "poincare_maps", "hydra_plus"]:
                    embedding_runner.train(adjacency_matrix=A, model_path=model_path)
                else:
                    embedding_runner.train(edge_list=edge_list, model_path=model_path)

                embeddings = embedding_runner.get_all_embeddings(model_path)

                # End timing
                end_time = time.time()
                elapsed_time = end_time - start_time
                timing_results[embedding_type].append(elapsed_time)

                print(f"    ✓ Run {run+1} completed: {elapsed_time:.2f}s")

                # Store embeddings from the first run for plotting
                if run == 0:
                    native_embedding_space = embedding_runner.model.native_space
                    all_embeddings_data.append(
                        {
                            "type": embedding_type,
                            "embeddings": embeddings,
                            "native_space": native_embedding_space,
                            "runner": embedding_runner,
                        }
                    )

            # Calculate and print statistics for this embedding type
            if native_embedding_space is not None:
                times = np.array(timing_results[embedding_type])
                mean_time = np.mean(times)
                std_time = np.std(times)
                print(f"\n✓ {embedding_type} completed (native space: {native_embedding_space})")
                print(f"  Mean time: {mean_time:.2f}s ± {std_time:.2f}s")
            else:
                print(f"\n✗ {embedding_type} failed - no successful runs")

    # Plot all embeddings in subplots
    print(f"\n{'='*60}")
    print(f"Generating comparison plot in {args.output_space} coordinates...")
    print(f"{'='*60}")

    # Mapping for embedding type names to display titles
    title_mapping = {
        "poincare_embeddings": "Poincare Embeddings",
        "lorentz": "Lorentz Embeddings",
        "poincare_maps": "Poincare Maps",
        "dmercator": "D-Mercator",
        "hydra": "Hydra",
        "hydra_plus": "Hydra+",
        "hypermap": "HyperMap",
    }

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
            title=title_mapping.get(data["type"], data["type"].replace("_", " ").title()),
            point_size=80,
            show_node_labels=False,
            node_label_size=6,
            edge_alpha=0.4,
            edge_width=0.8,
            colormap="viridis",
        )

    # Hide unused subplots
    for i in range(len(all_embeddings_data), 9):
        axes[i].axis("off")

    plt.tight_layout()

    plot_path = os.path.join(args.plot_dir, f"all_embeddings_comparison_{args.output_space}.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Comparison plot saved to {plot_path}")
    plt.close()

    # Print timing summary (excluding lorentz)
    if timing_results:
        print(f"\n{'='*60}")
        print(f"TIMING SUMMARY ({args.num_runs} runs per embedding type)")
        print(f"Note: lorentz model was loaded from disk (not trained)")
        print(f"{'='*60}")
        print(f"{'Embedding Type':<25} {'Mean ± Std (s)':<20}")
        print(f"{'-'*60}")

        # Calculate statistics and sort by mean time
        timing_stats = {}
        for embedding_type, times in timing_results.items():
            times_array = np.array(times)
            timing_stats[embedding_type] = {
                "mean": np.mean(times_array),
                "std": np.std(times_array),
                "min": np.min(times_array),
                "max": np.max(times_array),
            }

        sorted_timings = sorted(timing_stats.items(), key=lambda x: x[1]["mean"])

        for embedding_type, stats in sorted_timings:
            mean_s = stats["mean"]
            std_s = stats["std"]
            print(f"{embedding_type:<25} {mean_s:>8.2f} ± {std_s:<6.2f}")

        print(f"{'-'*60}")
        total_mean = sum(stats["mean"] for stats in timing_stats.values())
        print(f"{'TOTAL (mean)':<25} {total_mean:>18.2f}")
        print(f"{'='*60}")

        # Print detailed statistics
        print(f"\n{'='*80}")
        print("DETAILED TIMING STATISTICS")
        print(f"Note: lorentz model was loaded from disk (not trained)")
        print(f"{'='*80}")
        print(f"{'Embedding Type':<20} {'Mean (s)':<12} {'Std (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
        print(f"{'-'*80}")

        for embedding_type, stats in sorted_timings:
            print(f"{embedding_type:<20} {stats['mean']:>11.2f} {stats['std']:>11.2f} " f"{stats['min']:>11.2f} {stats['max']:>11.2f}")

        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
