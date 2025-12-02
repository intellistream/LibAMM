#!/bin/bash
# Setup script for bolt dependency
# This script downloads and sets up the bolt library for downstream inference benchmarks.
# Run this script only if you need to run the MADNESS evaluation benchmarks.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOLT_DIR="${SCRIPT_DIR}/bolt"
BOLT_REPO="https://github.com/dblalock/bolt.git"

echo "=== Bolt Setup Script ==="
echo "This will download the bolt library for downstream inference benchmarks."
echo ""

# Check if bolt already exists
if [ -d "${BOLT_DIR}" ]; then
    echo "bolt directory already exists at: ${BOLT_DIR}"
    read -p "Do you want to remove and re-download? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing bolt directory."
        exit 0
    fi
    echo "Removing existing bolt directory..."
    rm -rf "${BOLT_DIR}"
fi

# Clone bolt repository
echo "Cloning bolt from ${BOLT_REPO}..."
git clone --depth 1 "${BOLT_REPO}" "${BOLT_DIR}"

# Remove .git to make it a regular directory (not a submodule)
echo "Converting to regular directory (removing .git)..."
rm -rf "${BOLT_DIR}/.git"

# Initialize kmc2 submodule manually
echo "Setting up kmc2 dependency..."
KMC2_DIR="${BOLT_DIR}/third_party/kmc2"
if [ ! -d "${KMC2_DIR}" ] || [ -z "$(ls -A ${KMC2_DIR} 2>/dev/null)" ]; then
    echo "Downloading kmc2..."
    rm -rf "${KMC2_DIR}"
    git clone --depth 1 https://github.com/obachem/kmc2.git "${KMC2_DIR}"
    rm -rf "${KMC2_DIR}/.git"
fi

echo ""
echo "=== Setup Complete ==="
echo "bolt has been downloaded to: ${BOLT_DIR}"
echo ""
echo "To complete setup, run the following commands:"
echo ""
echo "  cd ${KMC2_DIR}"
echo "  pip install numpy==1.23.1 cython numba zstandard seaborn"
echo "  python3 setup.py build_ext --build-lib=."
echo ""
echo "Then you can run the benchmarks:"
echo ""
echo "  cd ${BOLT_DIR}/experiments"
echo "  python3 -m python.amm_main"
echo ""
