"""
Plot Gaussian hyperbolic embeddings in Poincaré disk.

This script:
1. Generates Gaussian embeddings using gaussian_loader.py
2. Plots the embeddings in Poincaré disk with class labels
"""

import argparse
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from gaussian_loader import GaussianLoader


def plot_embeddings_poincare(
    embeddings,
    labels,
    title="Gaussian Embeddings (Poincaré Disk)",
    point_size=50,
    show_node_labels=False,
    node_label_size=6,
    colormap="tab10",
    ax=None,
):
    """
    Plot embeddings in Poincaré disk.

    Parameters:
    -----------
    embeddings : np.ndarray
        Embeddings in Poincaré disk (N, 2)
    labels : np.ndarray
        Class labels (N,)
    title : str
        Plot title
    point_size : int
        Size of points
    show_node_labels : bool
        Whether to show node labels
    node_label_size : int
        Font size for node labels
    colormap : str
        Colormap name
    ax : matplotlib axis or None
        Axis to plot on (if None, creates new figure)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    x, y = embeddings[:, 0], embeddings[:, 1]

    ax.set_title(title, fontsize=14, pad=20)

    # Plot points by class
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
                linewidth=0.5,
                color=label_to_color[label],
                label=f"Class {label}",
                zorder=2,
                alpha=0.7,
            )
    else:
        ax.scatter(x, y, s=point_size, edgecolor="black", linewidth=0.5, color="skyblue", zorder=2, alpha=0.7)

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
    circle = plt.Circle((0, 0), 1, fill=False, color="gray", linewidth=2, zorder=0)
    ax.add_artist(circle)

    # Set limits and aspect
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add legend
    if labels is not None:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=10)

    return ax


def main():
    parser = argparse.ArgumentParser(description="Plot Gaussian hyperbolic embeddings in Poincaré disk")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1250,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=6,
        help="Number of classes/clusters",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=2,
        help="Dimension of the hyperbolic space",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset generation",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=2.0,
        help="Standard deviation of cluster noise",
    )
    parser.add_argument(
        "--separation",
        type=float,
        default=2.0,
        help="Separation between cluster centers",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="test/gaussian_plots",
        help="Directory to save the plots",
    )
    parser.add_argument(
        "--show_node_labels",
        action="store_true",
        help="Show node labels on the plot",
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

    # Create directory
    os.makedirs(args.plot_dir, exist_ok=True)

    print("=" * 80)
    print("PLOTTING GAUSSIAN EMBEDDINGS IN POINCARÉ DISK")
    print("=" * 80)
    print(f"\nDataset Configuration:")
    print(f"  Samples: {args.num_samples}")
    print(f"  Classes: {args.num_classes}")
    print(f"  Dimension: {args.dimension}")
    print(f"  Noise std: {args.noise_std}")
    print(f"  Separation: {args.separation}")
    print(f"  Seed: {args.seed}")

    # Generate Gaussian embeddings
    print("\n" + "-" * 80)
    print("STEP 1: Generating Gaussian Embeddings")
    print("-" * 80)

    loader = GaussianLoader()
    embeddings_poincare, labels = loader.generate_embeddings_and_labels(
        dimension=args.dimension,
        num_samples=args.num_samples,
        num_classes=args.num_classes,
        seed=args.seed,
        noise_std=args.noise_std,
        separation=args.separation,
        convert_to_poincare=True,
    )

    print(f"\n✓ Embeddings generated!")
    print(f"  Shape: {embeddings_poincare.shape}")
    print(f"  Space: Poincaré disk")
    print(f"  Labels: {len(labels)} samples, {len(np.unique(labels))} classes")
    print(f"  X range: [{embeddings_poincare[:, 0].min():.4f}, {embeddings_poincare[:, 0].max():.4f}]")
    print(f"  Y range: [{embeddings_poincare[:, 1].min():.4f}, {embeddings_poincare[:, 1].max():.4f}]")

    # Plot embeddings
    print("\n" + "-" * 80)
    print("STEP 2: Plotting Embeddings")
    print("-" * 80)
    print(f"\nPlot Configuration:")
    print(f"  Point size: {args.point_size}")
    print(f"  Show node labels: {args.show_node_labels}")
    print(f"  Colormap: {args.colormap}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot embeddings
    plot_embeddings_poincare(
        embeddings=embeddings_poincare,
        labels=labels,
        title=f"Gaussian Embeddings (Poincaré Disk)\n{args.num_samples} samples, {args.num_classes} classes",
        point_size=args.point_size,
        show_node_labels=args.show_node_labels,
        node_label_size=6,
        colormap=args.colormap,
        ax=ax,
    )

    plt.tight_layout()

    # Save plot
    plot_filename = f"gaussian_raw_embeddings_poincare_c{args.num_classes}_n{args.num_samples}.pdf"
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
    print(f"\nPlots saved to: {args.plot_dir}")


if __name__ == "__main__":
    main()
