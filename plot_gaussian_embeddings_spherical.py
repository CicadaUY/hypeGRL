"""
Plot Gaussian hyperbolic embeddings in spherical/polar coordinates.

This script:
1. Generates Gaussian embeddings in hyperboloid coordinates using gaussian_loader.py
2. Converts them from hyperboloid to Poincaré (using gaussian_loader's conversion)
3. Converts them from Poincaré to spherical/polar coordinates (with full circle angles)
4. Plots the embeddings in polar coordinates with class labels

Note: Uses a custom 2D conversion that properly handles angles in the full circle [0, 2π].
"""

import argparse
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from gaussian_loader import GaussianLoader
from utils.geometric_conversions import HyperbolicConversions


def poincare_to_spherical_2d(poincare_coords: np.ndarray) -> np.ndarray:
    """
    Convert 2D Poincaré coordinates to spherical/polar coordinates.

    Parameters:
    -----------
    poincare_coords : np.ndarray
        Coordinates in Poincaré disk (N, 2)

    Returns:
    --------
    spherical_coords : np.ndarray
        Spherical coordinates (N, 2) where [:, 0] is radius and [:, 1] is angle
    """
    x, y = poincare_coords[:, 0], poincare_coords[:, 1]

    # Compute Euclidean radius in Poincaré disk
    r_euclidean = np.sqrt(x**2 + y**2)

    # Convert to hyperbolic radius using the Poincaré disk metric
    # r_hyperbolic = 2 * arctanh(r_euclidean)
    # But to avoid numerical issues near the boundary, use the formula:
    # r_hyperbolic = ln((1 + r_euclidean) / (1 - r_euclidean))
    r_hyperbolic = np.log((1 + r_euclidean) / (1 - r_euclidean))

    # Compute angle using atan2 to get full circle [-π, π]
    theta = np.arctan2(y, x)

    # Convert to [0, 2π] range
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)

    # Stack radius and angle
    spherical = np.stack([r_hyperbolic, theta], axis=1)

    return spherical


def plot_embeddings_spherical(
    embeddings,
    labels,
    title="Gaussian Embeddings (Spherical Coordinates)",
    point_size=50,
    show_node_labels=False,
    node_label_size=6,
    colormap="tab10",
    ax=None,
):
    """
    Plot embeddings in spherical coordinates on a polar axis.

    Parameters:
    -----------
    embeddings : np.ndarray
        Embeddings in spherical coordinates (N, 2) where [:, 0] is radius and [:, 1] is angle
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
        Polar axis to plot on (if None, creates new figure)
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="polar")

    r, theta = embeddings[:, 0], embeddings[:, 1]

    ax.set_title(title, fontsize=14, pad=20)

    # Calculate boundary radius for the plot
    max_radius = np.max(r) if len(r) > 0 else 1.0
    boundary_radius = max_radius * 1.1

    # Plot points by class
    if labels is not None:
        labels_arr = np.array(labels)
        unique_labels = sorted(set(labels_arr))
        cmap_obj = cm.get_cmap(colormap, len(unique_labels))
        label_to_color = {label: cmap_obj(i) for i, label in enumerate(unique_labels)}

        for label in unique_labels:
            indices = np.where(labels_arr == label)[0]
            ax.scatter(
                theta[indices],
                r[indices],
                s=point_size,
                edgecolor="black",
                linewidth=0.5,
                color=label_to_color[label],
                label=f"Class {label}",
                zorder=2,
                alpha=0.7,
            )
    else:
        ax.scatter(theta, r, s=point_size, edgecolor="black", linewidth=0.5, color="skyblue", zorder=2, alpha=0.7)

    # Add node labels
    if show_node_labels:
        for i in range(len(r)):
            ax.text(
                theta[i],
                r[i],
                str(i),
                fontsize=node_label_size,
                ha="center",
                va="center",
                zorder=3,
            )

    # Set limits
    ax.set_ylim(0, boundary_radius)
    ax.set_rlim(0, boundary_radius)

    # Add legend
    if labels is not None:
        ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0), fontsize=10)

    return ax


def main():
    parser = argparse.ArgumentParser(description="Plot Gaussian hyperbolic embeddings in spherical coordinates")
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
    print("PLOTTING GAUSSIAN EMBEDDINGS IN SPHERICAL COORDINATES")
    print("=" * 80)
    print(f"\nDataset Configuration:")
    print(f"  Samples: {args.num_samples}")
    print(f"  Classes: {args.num_classes}")
    print(f"  Dimension: {args.dimension}")
    print(f"  Noise std: {args.noise_std}")
    print(f"  Separation: {args.separation}")
    print(f"  Seed: {args.seed}")

    # Generate Gaussian embeddings in hyperboloid
    print("\n" + "-" * 80)
    print("STEP 1: Generating Gaussian Embeddings (Hyperboloid)")
    print("-" * 80)

    loader = GaussianLoader()
    embeddings_hyperboloid, labels = loader.generate_embeddings_and_labels(
        dimension=args.dimension,
        num_samples=args.num_samples,
        num_classes=args.num_classes,
        seed=args.seed,
        noise_std=args.noise_std,
        separation=args.separation,
        convert_to_poincare=False,
    )

    print(f"\n✓ Embeddings generated!")
    print(f"  Shape: {embeddings_hyperboloid.shape}")
    print(f"  Space: Hyperboloid")
    print(f"  Labels: {len(labels)} samples, {len(np.unique(labels))} classes")
    print(f"  Coordinate ranges:")
    for i in range(embeddings_hyperboloid.shape[1]):
        print(f"    Dim {i}: [{embeddings_hyperboloid[:, i].min():.4f}, {embeddings_hyperboloid[:, i].max():.4f}]")

    # Convert hyperboloid to Poincaré (gaussian_loader uses [t, x, y] format)
    print("\n" + "-" * 80)
    print("STEP 2: Converting Hyperboloid to Poincaré")
    print("-" * 80)

    # Use the gaussian_loader's own conversion function
    embeddings_poincare = loader._hyperboloid_to_poincare(embeddings_hyperboloid)

    print(f"\n✓ Converted to Poincaré!")
    print(f"  Shape: {embeddings_poincare.shape}")
    print(f"  X range: [{embeddings_poincare[:, 0].min():.4f}, {embeddings_poincare[:, 0].max():.4f}]")
    print(f"  Y range: [{embeddings_poincare[:, 1].min():.4f}, {embeddings_poincare[:, 1].max():.4f}]")

    # Convert to spherical coordinates using custom 2D conversion
    print("\n" + "-" * 80)
    print("STEP 3: Converting Poincaré to Spherical (2D Polar)")
    print("-" * 80)

    embeddings_spherical = poincare_to_spherical_2d(embeddings_poincare)

    print(f"\n✓ Converted to spherical coordinates!")
    print(f"  Shape: {embeddings_spherical.shape}")
    print(f"  Radius range: [{embeddings_spherical[:, 0].min():.4f}, {embeddings_spherical[:, 0].max():.4f}]")
    print(f"  Angle range (radians): [{embeddings_spherical[:, 1].min():.4f}, {embeddings_spherical[:, 1].max():.4f}]")
    print(
        f"  Angle range (degrees): [{np.rad2deg(embeddings_spherical[:, 1].min()):.1f}°, {np.rad2deg(embeddings_spherical[:, 1].max()):.1f}°]"
    )

    # Plot embeddings
    print("\n" + "-" * 80)
    print("STEP 4: Plotting Embeddings")
    print("-" * 80)
    print(f"\nPlot Configuration:")
    print(f"  Point size: {args.point_size}")
    print(f"  Show node labels: {args.show_node_labels}")
    print(f"  Colormap: {args.colormap}")

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="polar")

    # Plot embeddings
    plot_embeddings_spherical(
        embeddings=embeddings_spherical,
        labels=labels,
        title=f"Gaussian Embeddings (Spherical Coordinates)\n{args.num_samples} samples, {args.num_classes} classes",
        point_size=args.point_size,
        show_node_labels=args.show_node_labels,
        node_label_size=6,
        colormap=args.colormap,
        ax=ax,
    )

    plt.tight_layout()

    # Save plot
    plot_filename = f"gaussian_raw_embeddings_spherical_c{args.num_classes}_n{args.num_samples}.pdf"
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
