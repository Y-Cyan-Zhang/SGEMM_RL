#pragma once

// Common constants used by kernels and runner
#define WARPSIZE 32

// Kernel 12: Double Buffering with async memcpy (cuda::barrier)
#include "12_kernel_double_buffering.cuh"
