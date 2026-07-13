#!/bin/bash

# Gaussian Hyperbolic KNN Experiments
# This script demonstrates different ways to use the Gaussian dataset
# All experiments use the default parameters from run_gaussian_benchmark.py

echo "=========================================="
echo "Gaussian Hyperbolic KNN Experiment Suite"
echo "=========================================="
echo ""

# Experiment 1: Basic 2D Gaussian with default parameters (1250 samples, 2 classes)
echo "Experiment 1: Basic 2D Gaussian Classification"
echo "------------------------------------------"
python -m test.other.hyperbolic_knn \
    --dataset gaussian \
    --embedding_type "hydra_plus" \
    --gaussian_num_samples 1250 \
    --gaussian_num_classes 2 \
    --gaussian_dimension 2 \
    --dim 2 \
    --n_iterations 3 \
    --k_values 3 5 7

echo ""
echo ""

# Experiment 2: Higher dimensional (4D) with same sample count
echo "Experiment 2: 4D Gaussian with 2 Classes"
echo "------------------------------------------"
python -m test.other.hyperbolic_knn \
    --dataset gaussian \
    --embedding_type "hydra_plus" \
    --gaussian_num_samples 1250 \
    --gaussian_num_classes 2 \
    --gaussian_dimension 4 \
    --dim 4 \
    --n_iterations 3 \
    --k_values 5 7 10

echo ""
echo ""

# Experiment 3: Large scale 8D dataset
echo "Experiment 3: 8D Gaussian"
echo "------------------------------------------"
python -m test.other.hyperbolic_knn \
    --dataset gaussian \
    --embedding_type "hydra_plus" \
    --gaussian_num_samples 1250 \
    --gaussian_num_classes 2 \
    --gaussian_dimension 8 \
    --dim 8 \
    --n_iterations 3 \
    --k_values 5 10 15

echo ""
echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in: test/other/results/gaussian/"
echo "=========================================="
