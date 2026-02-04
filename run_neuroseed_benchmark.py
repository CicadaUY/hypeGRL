"""
NeuroSEED Hyperbolic Dataset Benchmark Script

This script runs hyperbolic decision tree benchmarks on NeuroSEED hyperbolic embeddings
using the neuroseed_loader module and hyperdt benchmarks.
"""

import os
from time import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from hyperdt.ensemble import HyperbolicRandomForestClassifier
from hyperdt.tree import HyperbolicDecisionTreeClassifier
from neuroseed_loader import get_space, get_testing_data, get_training_data

# ============================================================================
# Benchmark Configuration
# ============================================================================

# Dataset parameters
dims = [2, 4, 8, 16]  # Dimensionality of hyperbolic space
seeds = [0, 1, 2, 3, 4]  # Random seeds for multiple runs
n_samples_total = 1250  # Total number of samples (including train/test split)

# Tree/Forest hyperparameters
max_depth = 3
num_classifiers = 10  # Number of trees in the Random Forest
min_samples_leaf = 1


# ============================================================================
# Evaluation Function
# ============================================================================


def evaluate_neuroseed_hdt(dimension, seed, num_samples, params):
    """
    Evaluate hyperbolic decision trees/forests on NeuroSEED dataset.

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
    print(f"\nLoading NeuroSEED dataset (dim={dimension}, seed={seed}, n={num_samples})...")

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
    print(f"Number of classes: {len(np.unique(y_train))}")

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


def run_benchmark(output_dir="test/other/results/neuroseed"):
    """
    Run the full benchmark and save results.

    Parameters:
    -----------
    output_dir : str
        Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("NEUROSEED HYPERBOLIC EMBEDDINGS BENCHMARK")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Classifiers: Hyperbolic Decision Tree & Hyperbolic Random Forest")
    print(f"  Dimensions: {dims}")
    print(f"  Seeds: {seeds}")
    print(f"  Total samples: {n_samples_total} (80% train / 20% test)")
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

    total_runs = len(dims) * len(seeds)
    current_run = 0

    # Main benchmark loop
    for dim in dims:
        for seed in seeds:
            current_run += 1
            print("\n" + "=" * 80)
            print(f"RUN {current_run}/{total_runs}: dim={dim}, seed={seed}")
            print("=" * 80)

            # Run evaluation
            try:
                f1_scores_tree, f1_scores_forest, tree_time, forest_time, init_time = evaluate_neuroseed_hdt(
                    dimension=dim,
                    seed=seed,
                    num_samples=n_samples_total,
                    params=params,
                )

                # Save results to dataframe - Decision Tree
                for fold, score in enumerate(f1_scores_tree):
                    results.loc[len(results)] = [
                        int(n_samples_total * 0.8),  # Training samples
                        "neuroseed",
                        dim,
                        seed,
                        "tree",
                        fold,
                        score,
                    ]

                # Save results to dataframe - Random Forest
                for fold, score in enumerate(f1_scores_forest):
                    results.loc[len(results)] = [
                        int(n_samples_total * 0.8),  # Training samples
                        "neuroseed",
                        dim,
                        seed,
                        "forest",
                        fold,
                        score,
                    ]

                times.loc[len(times)] = [
                    int(n_samples_total * 0.8),
                    "neuroseed",
                    dim,
                    seed,
                    "tree",
                    tree_time,
                    init_time,
                ]

                times.loc[len(times)] = [
                    int(n_samples_total * 0.8),
                    "neuroseed",
                    dim,
                    seed,
                    "forest",
                    forest_time,
                    init_time,
                ]

            except Exception as e:
                print(f"\n!!! ERROR in run {current_run}: {e}")
                import traceback

                traceback.print_exc()
                continue

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
    # Save results to text file
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

    # Default output file path
    output_file = os.path.join(output_dir, "neuroseed_hyperbolic_rf_benchmark_results.txt")

    with open(output_file, "w") as f:
        # Write header with experiment configuration
        f.write("=" * 90 + "\n")
        f.write("NEUROSEED HYPERBOLIC DECISION TREE/FOREST BENCHMARK RESULTS\n")
        f.write("=" * 90 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("CONFIGURATION:\n")
        f.write("-" * 90 + "\n")
        f.write(f"Dataset: NeuroSEED (American Gut Microbiome) Hyperbolic Embeddings\n")
        f.write(f"Classifiers: Hyperbolic Decision Tree & Hyperbolic Random Forest\n")
        f.write(f"Number of Estimators (Forest): {num_classifiers}\n")
        f.write(f"Max Depth: {max_depth}\n")
        f.write(f"Min Samples Leaf: {min_samples_leaf}\n")
        f.write(f"Dimensions Tested: {dims}\n")
        f.write(f"Random Seeds: {seeds}\n")
        f.write(f"Number of Iterations: {len(seeds)}\n")
        f.write(f"Training Samples per Run: {int(n_samples_total * 0.8)}\n")
        f.write(f"Total Samples (before split): {n_samples_total}\n")
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

    # Also save raw dataframes
    results_csv = os.path.join(output_dir, "neuroseed_results.csv")
    times_csv = os.path.join(output_dir, "neuroseed_times.csv")
    results.to_csv(results_csv, index=False)
    times.to_csv(times_csv, index=False)
    print(f"✓ Raw results saved to: {results_csv}")
    print(f"✓ Timing data saved to: {times_csv}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NeuroSEED hyperbolic embeddings benchmark")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test/other/results/neuroseed",
        help="Directory to save results",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="Dimensions to test",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1250,
        help="Total number of samples to use (before train/test split)",
    )

    args = parser.parse_args()

    # Update global variables
    dims = args.dims
    n_samples_total = args.num_samples

    run_benchmark(output_dir=args.output_dir)
