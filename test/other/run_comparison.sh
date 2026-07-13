#!/bin/bash

# Script to compare dmercator vs poincare_maps embeddings for gaussian and polblogs datasets

echo "=========================================="
echo "Running Embedding Comparisons"
echo "=========================================="
echo ""

# Default parameters
OUTPUT_SPACE="poincare"
SHOW_EDGES=""
SHOW_LEGEND=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_space)
            OUTPUT_SPACE="$2"
            shift 2
            ;;
        --show_edges)
            SHOW_EDGES="--show_edges"
            shift
            ;;
        --show_legend)
            SHOW_LEGEND="--show_legend"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--output_space SPACE] [--show_edges] [--show_legend]"
            exit 1
            ;;
    esac
done

# Run comparison for Gaussian dataset
echo "=========================================="
echo "1. Gaussian Dataset Comparison"
echo "=========================================="
python -m test.other.compare_embeddings \
    --dataset gaussian \
    --output_space $OUTPUT_SPACE \
    --gaussian_num_samples 1250 \
    --gaussian_num_classes 6 \
    --gaussian_dimension 2 \
    --k_neighbors 10 \
    --gaussian_seed 42 \
    $SHOW_EDGES \
    $SHOW_LEGEND

echo ""
echo "=========================================="
echo "2. PolBlogs Dataset Comparison"
echo "=========================================="
python -m test.other.compare_embeddings \
    --dataset polblogs \
    --output_space $OUTPUT_SPACE \
    $SHOW_EDGES \
    $SHOW_LEGEND

echo ""
echo "=========================================="
echo "All comparisons completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - test/other/plots/embeddings/gaussian_comparison_${OUTPUT_SPACE}.pdf"
echo "  - test/other/plots/embeddings/polblogs_comparison_${OUTPUT_SPACE}.pdf"
echo ""
