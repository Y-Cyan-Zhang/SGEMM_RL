#!/bin/bash
# Build and run on Google Colab T4
# Usage: !bash colab_build_and_run.sh

set -e

# Print GPU info
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

# Clean build
rm -rf build
mkdir build
cd build

# Configure for T4 (SM 7.5) 
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

echo ""
echo "=== Running cuBLAS baseline ==="
./sgemm 0

echo ""
echo "=== Running optimized double-buffering kernel ==="
./sgemm 12
