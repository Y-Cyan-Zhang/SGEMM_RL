#!/bin/bash
# Build and run on Google Colab T4
# Usage: bash build_and_run.sh
set -e

rm -rf build
mkdir build
cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=75 ..
make -j$(nproc)

echo ""
echo "=== Running cuBLAS baseline ==="
./sgemm 0

echo ""
echo "=== Running optimized double-buffering kernel ==="
./sgemm 12
