#!/bin/bash
# Script to check Python code formatting locally

set -e

echo "Installing formatting tools..."
pip install -q black ruff

echo ""
echo "=== Checking code formatting with Black ==="
black --check --diff kernels/ layout_plot/ experiments/ scripts/

echo ""
echo "=== Linting with Ruff ==="
ruff check kernels/ layout_plot/ experiments/ scripts/

echo ""
echo "✅ All formatting checks passed!"
