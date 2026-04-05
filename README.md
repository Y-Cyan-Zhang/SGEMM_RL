# Performance Results

## Wafer First Run

Benchmarked on [GPU MODEL] with matrix dimensions m=n=k.

### GFLOPS Comparison

| Matrix Size | cuBLAS | Optimized Kernel | % of cuBLAS |
|-------------|--------|------------------|-------------|
| 128         | 2.9    | 43.2             | 1489%       |
| 256         | 35.5   | 197.8            | 557%        |
| 512         | 1968.0 | 860.4            | 44%         |
| 1024        | 2587.3 | 1774.2           | 69%         |
| 2048        | 3463.7 | 3482.3           | 101%        |
| 4096        | 4216.5 | 3972.7           | 94%         |

### Key Observations

- Outperforms cuBLAS on small matrices (128, 256)
- Reaches **94-101%** of cuBLAS performance at large sizes (2048+)
- Double-buffering kernel optimized via reinforcement learning

### Raw Timing (seconds)

| Size | cuBLAS   | Optimized |
|------|----------|-----------|
| 128  | 0.001454 | 0.000097  |
| 256  | 0.000944 | 0.000170  |
| 512  | 0.000136 | 0.000312  |
| 1024 | 0.000830 | 0.001210  |
| 2048 | 0.004960 | 0.004933  |
| 4096 | 0.032595 | 0.034596  |

## Wafer Second Run

Benchmarked on [GPU MODEL] with matrix dimensions m=n=k, alpha=0.5, beta=3.

### GFLOPS Comparison

| Matrix Size | cuBLAS | Optimized Kernel | % of cuBLAS |
|-------------|--------|------------------|-------------|
| 128         | 27.3   | 52.0             | 190%        |
| 256         | 112.1  | 222.7            | 199%        |
| 512         | 1972.5 | 924.4            | 47%         |
| 1024        | 2587.8 | 1747.6           | 68%         |
| 2048        | 3391.3 | 3213.0           | 95%         |
| 4096        | 4107.4 | 3820.8           | 93%         |

### Key Observations

- Outperforms cuBLAS on small matrices (128, 256) by ~2x
- Reaches **93-95%** of cuBLAS performance at large sizes (2048+)
- Double-buffering kernel optimized via reinforcement learning

### Raw Timing (seconds)

| Size | cuBLAS   | Optimized |
|------|----------|-----------|
| 128  | 0.000154 | 0.000081  |
| 256  | 0.000299 | 0.000151  |
| 512  | 0.000136 | 0.000290  |
| 1024 | 0.000830 | 0.001229  |
| 2048 | 0.005066 | 0.005347  |
| 4096 | 0.033461 | 0.035971  |