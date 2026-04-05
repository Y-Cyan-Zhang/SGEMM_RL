#pragma once

// Common constants used by kernels and runner
#define WARPSIZE 32

// Kernel 12: Double Buffering with __syncthreads (sm_75 compatible)
#include "12_kernel_double_buffering.cuh"
