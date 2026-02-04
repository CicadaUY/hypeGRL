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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

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

# Classifiers to benchmark
clf_names = ["hrf", "rf"]  # "hrf" = Hyperbolic RF, "rf" = Euclidean RF

# Dataset parameters
dims = [2, 4, 8, 16]  # Dimensionality of hyperbolic space
seeds = [0, 1, 2, 3, 4]  # Random seeds for multiple runs
n_samples_train = [1000]  # Number of training samples

# Tree hyperparameters
max_depth = 3
num_classifiers = 10  # Number of trees in the forest (use 1 for single decision tree)
min_samples_leaf = 1

# Adjust for train_test split
n_samples = [int(x / 0.8) for x in n_samples_train]


# ============================================================================
# Evaluation Function
# ============================================================================


def evaluate_gaussian_hdt(dimension, seed, num_samples, params):
    """
    Evaluate hyperbolic decision trees on Gaussian dataset.

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
    f1_scores_hrf : list
        F1 scores for Hyperbolic RF across folds
    f1_scores_rf : list
        F1 scores for Euclidean RF across folds
    hrf_time : float
        Time taken for HRF evaluation
    rf_time : float
        Time taken for RF evaluation
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
    args = {
        "n_estimators": params["num_trees"],
        "max_depth": params["max_depth"],
        "min_samples_leaf": params["min_samples_leaf"],
        "random_state": seed,
    }
    use_tree = False
    if args["n_estimators"] == 1:
        del args["n_estimators"]  # This is a decision tree now
        del args["random_state"]
        use_tree = True

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    iterator = list(kf.split(X_train))

    t1 = time()

    # ========================================
    # Hyperbolic RF/Tree
    # ========================================
    f1_scores_hrf = []
    if "hrf" in clf_names:
        print("\nEvaluating Hyperbolic RF/Tree...")
        for fold_idx, (train_index, test_index) in enumerate(iterator):
            print(f"  Fold {fold_idx + 1}/5...", end=" ")
            try:
                if use_tree:
                    hrf = HyperbolicDecisionTreeClassifier(**args)
                    hrf.fit(X_train[train_index], y_train[train_index])
                else:
                    hrf = HyperbolicRandomForestClassifier(**args)
                    hrf.fit(X_train[train_index], y_train[train_index])

                y_pred = hrf.predict(X_train[test_index])
                f1 = f1_score(y_train[test_index], y_pred, average="micro")
                f1_scores_hrf.append(f1)
                print(f"F1: {f1:.4f}")
            except Exception as e:
                print(f"Error: {e}")
                f1_scores_hrf.append(np.nan)

    t2 = time()

    # ========================================
    # Euclidean RF/Tree
    # ========================================
    f1_scores_rf = []
    if "rf" in clf_names:
        print("\nEvaluating Euclidean RF/Tree...")
        for fold_idx, (train_index, test_index) in enumerate(iterator):
            print(f"  Fold {fold_idx + 1}/5...", end=" ")
            try:
                if use_tree:
                    rf = DecisionTreeClassifier(**args)
                else:
                    rf = RandomForestClassifier(**args)
                rf.fit(X_train[train_index], y_train[train_index])
                y_pred = rf.predict(X_train[test_index])
                f1 = f1_score(y_train[test_index], y_pred, average="micro")
                f1_scores_rf.append(f1)
                print(f"F1: {f1:.4f}")
            except Exception as e:
                print(f"Error: {e}")
                f1_scores_rf.append(np.nan)

    t3 = time()

    return f1_scores_hrf, f1_scores_rf, t2 - t1, t3 - t2, t1 - t0


# ============================================================================
# Main Benchmark Loop
# ============================================================================


def run_benchmark(output_dir="results/gaussian_benchmark"):
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
    print(f"  Dimensions: {dims}")
    print(f"  Seeds: {seeds}")
    print(f"  Training samples: {n_samples_train}")
    print(f"  Total samples: {n_samples}")
    print(f"  Max depth: {max_depth}")
    print(f"  Number of trees: {num_classifiers}")
    print(f"  Min samples leaf: {min_samples_leaf}")
    print(f"  Classifiers: {clf_names}")

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
                f1_scores_hrf, f1_scores_rf, hrf_time, rf_time, init_time = evaluate_gaussian_hdt(
                    dimension=dim,
                    seed=seed,
                    num_samples=n,
                    params=params,
                )

                # Save results to dataframe
                scores = [f1_scores_hrf, f1_scores_rf]
                ts = [hrf_time, rf_time]

                for score_list, t, name in zip(scores, ts, clf_names):
                    for fold, score in enumerate(score_list):
                        results.loc[len(results)] = [
                            int(n * 0.8),  # Training samples
                            "gaussian",
                            dim,
                            seed,
                            name,
                            fold,
                            score,
                        ]
                    times.loc[len(times)] = [
                        int(n * 0.8),
                        "gaussian",
                        dim,
                        seed,
                        name,
                        t,
                        init_time,
                    ]

                # Save intermediate results (in case of failure)
                results.to_csv(f"{output_dir}/gaussian_results.csv", index=False)
                times.to_csv(f"{output_dir}/gaussian_times.csv", index=False)

                print(f"\nResults saved to {output_dir}/")

    # ========================================
    # Print Summary
    # ========================================
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE - SUMMARY")
    print("=" * 80)

    # Group by classifier and compute mean/std
    summary = results.groupby(["dim", "clf"])["f1_micro"].agg(["mean", "std", "count"])
    print("\nF1 Micro Scores (mean ± std):")
    print(summary)

    # Save summary
    summary.to_csv(f"{output_dir}/gaussian_summary.csv")

    print(f"\n✓ All results saved to {output_dir}/")
    print(f"  - gaussian_results.csv (detailed results)")
    print(f"  - gaussian_times.csv (timing information)")
    print(f"  - gaussian_summary.csv (summary statistics)")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    run_benchmark()
