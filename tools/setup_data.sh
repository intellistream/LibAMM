#!/bin/bash
# LibAMM Data Setup Script
# Creates symbolic links to SAGE data repository

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LIBAMM_ROOT="$SCRIPT_DIR/.."

echo "🔧 LibAMM Data Setup"
echo "===================="

# Detect SAGE_DATA_ROOT
if [ -n "$SAGE_DATA_ROOT" ]; then
    DATA_ROOT="$SAGE_DATA_ROOT"
elif [ -d "$LIBAMM_ROOT/../../../../../sage-benchmark/src/sage/data" ]; then
    DATA_ROOT="$LIBAMM_ROOT/../../../../../sage-benchmark/src/sage/data"
else
    echo "❌ Error: Cannot find SAGE data directory"
    echo "Please set SAGE_DATA_ROOT environment variable or ensure sage-benchmark is installed"
    exit 1
fi

echo "📁 Data root: $DATA_ROOT"

cd "$LIBAMM_ROOT"

# Create necessary directories
mkdir -p benchmark
mkdir -p test/torchscripts/VQ

# Link models (for downstream benchmarks)
if [ -d "$DATA_ROOT/libamm-benchmark/models" ]; then
    echo "🔗 Linking models..."
    rm -f benchmark/models  # Remove existing link/dir to prevent nested links
    ln -sf "$DATA_ROOT/libamm-benchmark/models" benchmark/models
fi

# Link test data
if [ -d "$DATA_ROOT/libamm-benchmark/test-data" ]; then
    echo "🔗 Linking test data..."
    rm -f test/torchscripts/VQ/data  # Remove existing link/dir to prevent nested links
    ln -sf "$DATA_ROOT/libamm-benchmark/test-data" test/torchscripts/VQ/data
    # Note: Individual .txt file symlinks are tracked in Git with relative paths
    # Don't recreate them here to avoid Git dirty state
    # They are already committed in the repository
fi

# Link datasets if available
if [ -d "$DATA_ROOT/libamm-benchmark/datasets" ]; then
    echo "🔗 Linking datasets..."
    rm -f benchmark/datasets  # Remove existing link/dir to prevent nested links
    ln -sf "$DATA_ROOT/libamm-benchmark/datasets" benchmark/datasets
fi

echo "✅ LibAMM data setup complete!"
echo ""
echo "Data links created:"
[ -L "benchmark/models" ] && echo "  ✓ benchmark/models -> $DATA_ROOT/libamm-benchmark/models"
[ -L "benchmark/datasets" ] && echo "  ✓ benchmark/datasets -> $DATA_ROOT/libamm-benchmark/datasets"
[ -L "test/torchscripts/VQ/data" ] && echo "  ✓ test/torchscripts/VQ/data -> $DATA_ROOT/libamm-benchmark/test-data"
echo ""
echo "💡 Tip: Set SAGE_DATA_ROOT=/path/to/data to customize data location"
