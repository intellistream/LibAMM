#!/bin/bash
# Script to migrate LibAMM from PyTorch to Eigen
# This performs automated find-and-replace operations

set -e

echo "Starting LibAMM PyTorch to Eigen migration..."

# Backup original files
echo "Creating backup..."
git stash push -m "Pre-migration backup"

# Replace includes
echo "Replacing torch includes with EigenTensor..."
find include/ src/ -type f \( -name "*.h" -o -name "*.cpp" \) -exec sed -i \
    -e 's/#include <torch\/torch.h>/#include <Utils\/EigenTensor.h>/g' \
    -e 's/#include <torch\/script.h>//g' \
    {} \;

# Replace torch::Tensor with LibAMM::Tensor
echo "Replacing torch::Tensor with LibAMM::Tensor..."
find include/ src/ -type f \( -name "*.h" -o -name "*.cpp" \) -exec sed -i \
    's/torch::Tensor/LibAMM::Tensor/g' \
    {} \;

# Replace torch:: function calls with LibAMM::
echo "Replacing torch:: namespace with LibAMM::..."
find include/ src/ -type f \( -name "*.h" -o -name "*.cpp" \) -exec sed -i \
    -e 's/torch::matmul/LibAMM::matmul/g' \
    -e 's/torch::zeros/LibAMM::zeros/g' \
    -e 's/torch::ones/LibAMM::ones/g' \
    -e 's/torch::rand/LibAMM::rand/g' \
    -e 's/torch::randint/LibAMM::randint/g' \
    -e 's/torch::arange/LibAMM::arange/g' \
    -e 's/torch::diag/LibAMM::diag/g' \
    -e 's/torch::sqrt/LibAMM::sqrt/g' \
    -e 's/torch::exp/LibAMM::exp/g' \
    -e 's/torch::clamp/LibAMM::clamp/g' \
    -e 's/torch::tensor/LibAMM::tensor/g' \
    {} \;

# Replace device specifiers
echo "Replacing CUDA device specifiers..."
find include/ src/ -type f \( -name "*.h" -o -name "*.cpp" \) -exec sed -i \
    -e 's/torch::kCUDA/LibAMM::kCUDA/g' \
    -e 's/torch::kCPU/LibAMM::kCPU/g' \
    -e 's/torch::kFloat32/"float32"/g' \
    {} \;

# Replace torch::Scalar
echo "Replacing torch::Scalar..."
find include/ src/ -type f \( -name "*.h" -o -name "*.cpp" \) -exec sed -i \
    's/torch::Scalar/LibAMM::Scalar/g' \
    {} \;

echo "Migration complete!"
echo "Please review the changes and fix any compilation errors manually."
echo "Common issues to check:"
echo "  - Tensor indexing syntax may need adjustment"
echo "  - Some advanced torch operations may not have direct equivalents"
echo "  - Check all TODO comments added during migration"
