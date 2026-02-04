#!/bin/bash

# Script to install hyperdt in the current virtual environment

echo "Installing hyperdt package..."
echo "Current directory: $(pwd)"
echo "Python executable: $(which python)"
echo ""

# Install hyperdt in editable mode
cd hyperdt
pip install -e .

echo ""
echo "✓ Installation complete!"
echo "You can now run: python run_gaussian_benchmark.py"
