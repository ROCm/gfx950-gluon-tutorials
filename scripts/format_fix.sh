#!/bin/bash
# Script to auto-fix Python code formatting

set -e

echo "Installing formatting tools..."
pip install -q black ruff

echo ""
echo "=== Formatting code with Black ==="
black kernels/ layout_plot/ experiments/ scripts/

echo ""
echo "=== Auto-fixing with Ruff ==="
ruff check --fix kernels/ layout_plot/ experiments/ scripts/

echo ""
echo "✅ Code formatting complete!"
