"""
Gaussian Hyperbolic Dataset Benchmark Script

This script runs hyperbolic decision tree benchmarks on Gaussian hyperbolic embeddings
using the gaussian_loader module and hyperdt benchmarks.
"""

import os
from time import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split

from gaussian_loader import GaussianLoader
from hyperdt.ensemble import HyperbolicRandomForestClassifier
from hyperdt.tree import HyperbolicDecisionTreeClassifier

# ============================================================================
# Data Loading Interface (compatible with hororf_benchmarks.py)
# ============================================================================


def get_training_data(class_label, seed, num_samples=1250, convert_to_poincare=True):
    """
    Get training data from Gaussian hyperbolic embeddings.

    Parameters:
    -----------
    class_label : int
        Dimensionality of the hyperbolic space (serves as the "class label")
    seed : int
        Random seed
    num_samples : int
        Total number of samples (before train/test split)
    convert_to_poincare : bool
        If True, returns data in Poincaré ball coordinates
        If False, returns data in hyperboloid coordinates

    Returns:
    --------
    data : torch.Tensor
        Training embeddings
    labels : torch.Tensor
        Training labels
    """
    loader = GaussianLoader()

    # Generate embeddings
    embeddings, labels = loader.generate_embeddings_and_labels(
        dimension=class_label,
        num_samples=num_samples,
        num_classes=6,
        seed=seed,
        noise_std=2.0,
        separation=2.0,
        convert_to_poincare=convert_to_poincare,
    )

    # Split train/test
    train_embeddings, _, train_labels, _ = train_test_split(embeddings, labels, test_size=0.2, random_state=seed)

    return torch.as_tensor(train_embeddings), torch.as_tensor(train_labels, dtype=int).flatten()


def get_testing_data(class_label, seed, num_samples=1250, convert_to_poincare=True):
    """
    Get testing data from Gaussian hyperbolic embeddings.

    Parameters:
    -----------
    class_label : int
        Dimensionality of the hyperbolic space (serves as the "class label")
    seed : int
        Random seed
    num_samples : int
        Total number of samples (before train/test split)
    convert_to_poincare : bool
        If True, returns data in Poincaré ball coordinates
        If False, returns data in hyperboloid coordinates

    Returns:
    --------
    data : torch.Tensor
        Testing embeddings
    labels : torch.Tensor
        Testing labels
    """
    loader = GaussianLoader()

    # Generate embeddings (same seed ensures same data generation)
    embeddings, labels = loader.generate_embeddings_and_labels(
        dimension=class_label,
        num_samples=num_samples,
        num_classes=2,
        seed=seed,
        noise_std=2.0,
        separation=2.0,
        convert_to_poincare=convert_to_poincare,
    )

    # Split train/test
    _, test_embeddings, _, test_labels = train_test_split(embeddings, labels, test_size=0.2, random_state=seed)

    return torch.as_tensor(test_embeddings), torch.as_tensor(test_labels, dtype=int).flatten()


def get_space():
    """Return the space type."""
    return "hyperbolic"


# ============================================================================
# Benchmark Configuration
# ============================================================================

# Dataset parameters
dims = [2, 4, 8, 16]  # Dimensionality of hyperbolic space
seeds = [0, 1, 2, 3, 4]  # Random seeds for multiple runs
n_samples_train = [1000]  # Number of training samples

# Tree/Forest hyperparameters
max_depth = 3
num_classifiers = 10  # Number of trees in the Random Forest
min_samples_leaf = 1

# Adjust for train_test split
n_samples = [int(x / 0.8) for x in n_samples_train]


# ============================================================================
# Evaluation Function
# ============================================================================


def evaluate_gaussian_hdt(dimension, seed, num_samples, params):
    """
    Evaluate hyperbolic decision trees/forests on Gaussian dataset.

    Parameters:
    -----------
    dimension : int
        Dimensionality of hyperbolic space
    seed : int
        Random seed
    num_samples : int
        Total number of samples
    params : dict
        Hyperparameters for the classifiers

    Returns:
    --------
    f1_scores_tree : list
        F1 scores for Hyperbolic Decision Tree across folds
    f1_scores_forest : list
        F1 scores for Hyperbolic Random Forest across folds
    tree_time : float
        Time taken for tree evaluation
    forest_time : float
        Time taken for forest evaluation
    init_time : float
        Time taken for data loading
    """
    # Dataset
    t0 = time()
    print(f"\nLoading Gaussian dataset (dim={dimension}, seed={seed}, n={num_samples})...")

    # Get data
    X_train, y_train = get_training_data(
        class_label=dimension,
        seed=seed,
        num_samples=num_samples,
        convert_to_poincare=False,  # Use hyperboloid coordinates
    )
    X_train = X_train.numpy()
    y_train = y_train.numpy()

    print(f"X shape: {X_train.shape}")
    print(f"y shape: {y_train.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")

    # Hyperparameters
    tree_args = {
        "max_depth": params["max_depth"],
        "min_samples_leaf": params["min_samples_leaf"],
    }

    forest_args = {
        "n_estimators": params["num_trees"],
        "max_depth": params["max_depth"],
        "min_samples_leaf": params["min_samples_leaf"],
        "random_state": seed,
    }

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    iterator = list(kf.split(X_train))

    t1 = time()

    # ========================================
    # Hyperbolic Decision Tree
    # ========================================
    f1_scores_tree = []
    print("\nEvaluating Hyperbolic Decision Tree...")
    for fold_idx, (train_index, test_index) in enumerate(iterator):
        print(f"  Fold {fold_idx + 1}/5...", end=" ")
        try:
            tree = HyperbolicDecisionTreeClassifier(**tree_args)
            tree.fit(X_train[train_index], y_train[train_index])
            y_pred = tree.predict(X_train[test_index])
            f1 = f1_score(y_train[test_index], y_pred, average="micro")
            f1_scores_tree.append(f1)
            print(f"F1: {f1:.4f}")
        except Exception as e:
            print(f"Error: {e}")
            f1_scores_tree.append(np.nan)

    t2 = time()

    # ========================================
    # Hyperbolic Random Forest
    # ========================================
    f1_scores_forest = []
    print("\nEvaluating Hyperbolic Random Forest...")
    for fold_idx, (train_index, test_index) in enumerate(iterator):
        print(f"  Fold {fold_idx + 1}/5...", end=" ")
        try:
            forest = HyperbolicRandomForestClassifier(**forest_args)
            forest.fit(X_train[train_index], y_train[train_index])
            y_pred = forest.predict(X_train[test_index])
            f1 = f1_score(y_train[test_index], y_pred, average="micro")
            f1_scores_forest.append(f1)
            print(f"F1: {f1:.4f}")
        except Exception as e:
            print(f"Error: {e}")
            f1_scores_forest.append(np.nan)

    t3 = time()

    return f1_scores_tree, f1_scores_forest, t2 - t1, t3 - t2, t1 - t0


# ============================================================================
# Main Benchmark Loop
# ============================================================================


def run_benchmark(output_dir="test/other/results/gaussian"):
    """
    Run the full benchmark and save results.

    Parameters:
    -----------
    output_dir : str
        Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("GAUSSIAN HYPERBOLIC EMBEDDINGS BENCHMARK")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Classifiers: Hyperbolic Decision Tree & Hyperbolic Random Forest")
    print(f"  Dimensions: {dims}")
    print(f"  Seeds: {seeds}")
    print(f"  Training samples: {n_samples_train}")
    print(f"  Total samples: {n_samples}")
    print(f"  Max depth: {max_depth}")
    print(f"  Number of estimators (forest): {num_classifiers}")
    print(f"  Min samples leaf: {min_samples_leaf}")

    # Initialize results dataframes
    results = pd.DataFrame(columns=["n_samples", "dataset", "dim", "seed", "clf", "fold", "f1_micro"])
    times = pd.DataFrame(columns=["n_samples", "dataset", "dim", "seed", "clf", "time", "init_time"])

    # Parameters for evaluation
    params = {
        "num_trees": num_classifiers,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
    }

    total_runs = len(n_samples) * len(dims) * len(seeds)
    current_run = 0

    # Main benchmark loop
    for n in n_samples:
        for dim in dims:
            for seed in seeds:
                current_run += 1
                print("\n" + "=" * 80)
                print(f"RUN {current_run}/{total_runs}: n_samples={int(n * 0.8)}, dim={dim}, seed={seed}")
                print("=" * 80)

                # Run evaluation
                f1_scores_tree, f1_scores_forest, tree_time, forest_time, init_time = evaluate_gaussian_hdt(
                    dimension=dim,
                    seed=seed,
                    num_samples=n,
                    params=params,
                )

                # Save results to dataframe - Decision Tree
                for fold, score in enumerate(f1_scores_tree):
                    results.loc[len(results)] = [
                        int(n * 0.8),  # Training samples
                        "gaussian",
                        dim,
                        seed,
                        "tree",
                        fold,
                        score,
                    ]

                # Save results to dataframe - Random Forest
                for fold, score in enumerate(f1_scores_forest):
                    results.loc[len(results)] = [
                        int(n * 0.8),  # Training samples
                        "gaussian",
                        dim,
                        seed,
                        "forest",
                        fold,
                        score,
                    ]

                times.loc[len(times)] = [
                    int(n * 0.8),
                    "gaussian",
                    dim,
                    seed,
                    "tree",
                    tree_time,
                    init_time,
                ]

                times.loc[len(times)] = [
                    int(n * 0.8),
                    "gaussian",
                    dim,
                    seed,
                    "forest",
                    forest_time,
                    init_time,
                ]

    # ========================================
    # Print Summary
    # ========================================
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE - SUMMARY")
    print("=" * 80)

    # Group by dimension and classifier
    print("\nHyperbolic Decision Tree F1 Micro Scores (mean ± std) by Dimension:")
    tree_summary = results[results["clf"] == "tree"].groupby("dim")["f1_micro"].agg(["mean", "std", "count"])
    print(tree_summary)

    print("\nHyperbolic Random Forest F1 Micro Scores (mean ± std) by Dimension:")
    forest_summary = results[results["clf"] == "forest"].groupby("dim")["f1_micro"].agg(["mean", "std", "count"])
    print(forest_summary)

    # ========================================
    # Save results to text file (following hyperbolic_knn.py format)
    # ========================================
    from datetime import datetime

    # Compute final results per dimension and classifier
    final_results_tree = {}
    final_results_forest = {}
    for dim in dims:
        tree_data = results[(results["dim"] == dim) & (results["clf"] == "tree")]
        forest_data = results[(results["dim"] == dim) & (results["clf"] == "forest")]

        final_results_tree[dim] = {
            "f1_mean": tree_data["f1_micro"].mean(),
            "f1_std": tree_data["f1_micro"].std(),
            "count": len(tree_data),
        }

        final_results_forest[dim] = {
            "f1_mean": forest_data["f1_micro"].mean(),
            "f1_std": forest_data["f1_micro"].std(),
            "count": len(forest_data),
        }

    # Find best dimension for each classifier
    best_dim_tree = max(final_results_tree.keys(), key=lambda d: final_results_tree[d]["f1_mean"])
    best_dim_forest = max(final_results_forest.keys(), key=lambda d: final_results_forest[d]["f1_mean"])

    # Default output file path (following hyperbolic_knn.py pattern)
    output_file = os.path.join(output_dir, "hyperbolic_rf_benchmark_results.txt")

    with open(output_file, "w") as f:
        # Write header with experiment configuration
        f.write("=" * 90 + "\n")
        f.write("HYPERBOLIC DECISION TREE/FOREST BENCHMARK RESULTS\n")
        f.write("=" * 90 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("CONFIGURATION:\n")
        f.write("-" * 90 + "\n")
        f.write(f"Dataset: Gaussian Hyperbolic Embeddings\n")
        f.write(f"Classifiers: Hyperbolic Decision Tree & Hyperbolic Random Forest\n")
        f.write(f"Number of Estimators (Forest): {num_classifiers}\n")
        f.write(f"Max Depth: {max_depth}\n")
        f.write(f"Min Samples Leaf: {min_samples_leaf}\n")
        f.write(f"Dimensions Tested: {dims}\n")
        f.write(f"Random Seeds: {seeds}\n")
        f.write(f"Number of Iterations: {len(seeds)}\n")
        f.write(f"Training Samples per Run: {n_samples_train[0]}\n")
        f.write(f"Total Samples (before split): {n_samples[0]}\n")
        f.write(f"Cross-validation Folds: 5\n")
        f.write(f"Total Runs: {total_runs}\n")
        f.write("\n")

        # Write summary results
        f.write("=" * 90 + "\n")
        f.write(f"SUMMARY OF RESULTS (Mean ± Std over {len(seeds)} seed(s))\n")
        f.write("=" * 90 + "\n\n")

        f.write("HYPERBOLIC DECISION TREE:\n")
        f.write(f"{'Dimension':<12} {'F1 Score':<20} {'Count':<8}\n")
        f.write("-" * 90 + "\n")
        for dim in sorted(final_results_tree.keys()):
            r = final_results_tree[dim]
            f.write(f"{dim:<12} " f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}      " f"{int(r['count']):<8}\n")
        f.write(
            f"\nBest dimension: {best_dim_tree} "
            f"(F1: {final_results_tree[best_dim_tree]['f1_mean']:.4f} ± "
            f"{final_results_tree[best_dim_tree]['f1_std']:.4f})\n\n"
        )

        f.write("\nHYPERBOLIC RANDOM FOREST:\n")
        f.write(f"{'Dimension':<12} {'F1 Score':<20} {'Count':<8}\n")
        f.write("-" * 90 + "\n")
        for dim in sorted(final_results_forest.keys()):
            r = final_results_forest[dim]
            f.write(f"{dim:<12} " f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}      " f"{int(r['count']):<8}\n")
        f.write(
            f"\nBest dimension: {best_dim_forest} "
            f"(F1: {final_results_forest[best_dim_forest]['f1_mean']:.4f} ± "
            f"{final_results_forest[best_dim_forest]['f1_std']:.4f})\n"
        )
        f.write("=" * 90 + "\n\n")

        # Write detailed iteration results if multiple seeds
        if len(seeds) > 1:
            f.write("=" * 90 + "\n")
            f.write("DETAILED ITERATION RESULTS BY DIMENSION\n")
            f.write("=" * 90 + "\n")

            for dim in sorted(dims):
                f.write(f"\n{'='*50}\n")
                f.write(f"DIMENSION = {dim}\n")
                f.write(f"{'='*50}\n\n")

                # Decision Tree results
                f.write("Hyperbolic Decision Tree:\n")
                tree_data = results[(results["dim"] == dim) & (results["clf"] == "tree")]
                for seed in seeds:
                    seed_data = tree_data[tree_data["seed"] == seed]
                    if len(seed_data) > 0:
                        f1_scores = seed_data["f1_micro"].values
                        f.write(f"  Seed {seed}: {f1_scores.tolist()}\n")
                f.write(f"  Mean ± Std: {tree_data['f1_micro'].mean():.4f} ± {tree_data['f1_micro'].std():.4f}\n\n")

                # Random Forest results
                f.write("Hyperbolic Random Forest:\n")
                forest_data = results[(results["dim"] == dim) & (results["clf"] == "forest")]
                for seed in seeds:
                    seed_data = forest_data[forest_data["seed"] == seed]
                    if len(seed_data) > 0:
                        f1_scores = seed_data["f1_micro"].values
                        f.write(f"  Seed {seed}: {f1_scores.tolist()}\n")
                f.write(f"  Mean ± Std: {forest_data['f1_micro'].mean():.4f} ± {forest_data['f1_micro'].std():.4f}\n")

    print(f"\n✓ Results saved to: {output_file}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    run_benchmark()
