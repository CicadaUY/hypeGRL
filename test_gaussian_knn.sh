#!/bin/bash

# Test script for running hyperbolic KNN with Gaussian dataset

echo "Testing Gaussian dataset with hyperbolic KNN..."
echo ""

# Run with gaussian dataset, minimal iterations for testing
# Using same defaults as run_gaussian_benchmark.py: 1250 samples, 2 classes
python -m test.other.hyperbolic_knn \
    --dataset gaussian \
    --embedding_type "hydra_plus" \
    --gaussian_num_samples 1250 \
    --gaussian_num_classes 2 \
    --gaussian_dimension 2 \
    --gaussian_seed 42 \
    --dim 2 \
    --n_iterations 2 \
    --k_values 3 5 7 \
    --k_neighbors 10

echo ""
echo "✓ Test complete!"
