#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/*
 * Highly-optimized SGEMM with double buffering for SM 7.5 (T4).
 *
 * Optimizations over the previous version:
 * ─────────────────────────────────────────
 * 1. 256 threads/block (8 warps) → 25% occupancy (2× vs 128-thread version).
 *    With 2 blocks/SM, the warp scheduler has 16 warps to interleave, much
 *    better for hiding both memory and ALU latency on T4's Turing architecture.
 *
 * 2. 64 accumulators/thread (was 128). Halving register pressure means the
 *    compiler can keep everything in registers without spilling. Estimated
 *    ~110 regs/thread → 256×110 = 28160 regs/block, 2 blocks = 56320 < 65536.
 *
 * 3. __forceinline__ on all device helpers to eliminate call overhead and let
 *    the compiler schedule instructions across load/compute boundaries.
 *
 * 4. Outer-product FMA structure: the `a` value is hoisted out of the inner
 *    N-loop, giving the compiler a clean FMA dependency chain per N-element.
 *
 * 5. Vectorized float4 loads for both A (with register scatter-transpose) and B.
 *
 * 6. __restrict__ on all pointers + --use_fast_math.
 *
 * 7. __launch_bounds__(256) for optimal register allocation by the compiler.
 *
 * No SMEM padding needed: within a warp, threads read consecutive m-indices
 * from As (stride-1), so all 32 threads hit different banks. The k-stride
 * of BM=128 (which is 0 mod 32) does NOT cause conflicts because all threads
 * in a warp read the SAME k-row in any given dotIdx iteration.
 *
 * Memory:
 *   SMEM = 2 × (128×16 + 16×128) × 4 = 32768 bytes = 32 KB → 2 blocks in 64 KB
 *
 * Config: BM=128, BN=128, BK=16, WM=64, WN=32,
 *         WNITER=2, TM=8, TN=4, 256 threads (8 warps)
 *   Warp layout: 2×4 (2 rows, 4 cols)
 *   WMITER=1, WSUBM=64, WSUBN=16
 *   64 results/thread, ~110 regs/thread
 */

namespace {

template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ __forceinline__ void
loadFromGmem(int N, int K,
             const float *__restrict__ A,
             const float *__restrict__ B,
             float *__restrict__ As,
             float *__restrict__ Bs,
             int innerRowA, int innerColA,
             int innerRowB, int innerColB) {
  // Load A: float4 along K dimension, scatter-transpose into As[k * BM + m]
  #pragma unroll
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  // Load B: float4 along N dimension (row-major, no transpose)
  #pragma unroll
  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ __forceinline__ void
processFromSmem(float *__restrict__ regM,
                float *__restrict__ regN,
                float *__restrict__ threadResults,
                const float *__restrict__ As,
                const float *__restrict__ Bs,
                const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  #pragma unroll
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // Load A tile column into registers
    #pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      #pragma unroll
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[dotIdx * BM + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    // Load B tile row into registers
    #pragma unroll
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      #pragma unroll
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[dotIdx * BN + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // Outer-product accumulate with hoisted `a` for FMA chains
    #pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      #pragma unroll
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        #pragma unroll
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          const float a = regM[wSubRowIdx * TM + resIdxM];
          #pragma unroll
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                a * regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    runSgemmDoubleBuffering2(int M, int N, int K, float alpha,
                             const float *__restrict__ A,
                             const float *__restrict__ B, float beta,
                             float *__restrict__ C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;

  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  // Double-buffered shared memory (no padding needed — see comment above)
  __shared__ float As[2 * BM * BK];
  __shared__ float Bs[2 * BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  float regM[WMITER * TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

  // Load first tile into buffer 0
  loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
      N, K, A, B, &As[0], &Bs[0],
      innerRowA, innerColA, innerRowB, innerColB);
  __syncthreads();

  // Main K-loop: all tiles except the last
  for (uint bkIdx = 0; bkIdx < K - BK; bkIdx += BK) {
    const int curBuf = (bkIdx / BK) & 1;
    const int nextBuf = 1 - curBuf;

    // Issue loads for next tile into alternate buffer (global→shared)
    loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A + BK, B + BK * N,
        &As[nextBuf * BM * BK], &Bs[nextBuf * BK * BN],
        innerRowA, innerColA, innerRowB, innerColB);

    // Compute current tile — compiler interleaves these with outstanding loads
    processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regM, regN, threadResults,
        &As[curBuf * BM * BK], &Bs[curBuf * BK * BN],
        warpRow, warpCol, threadRowInWarp, threadColInWarp);

    A += BK;
    B += BK * N;

    // Ensure next-tile loads complete and current-tile reads are done
    __syncthreads();
  }

  // Process the last tile
  {
    const int lastBuf = ((K / BK - 1)) & 1;
    processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regM, regN, threadResults,
        &As[lastBuf * BM * BK], &Bs[lastBuf * BK * BN],
        warpRow, warpCol, threadRowInWarp, threadColInWarp);
  }

  // Write results back with alpha/beta scaling (vectorized float4 stores)
  #pragma unroll
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    #pragma unroll
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      #pragma unroll
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        #pragma unroll
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}
