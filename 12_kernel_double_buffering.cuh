#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/*
 * Optimized SGEMM with double buffering for SM 7.5 (T4).
 *
 * Key optimizations vs the original SM 8.0+ version:
 * 1. Replaced cuda::barrier / cuda::memcpy_async with __syncthreads() and
 *    manual vectorized loads (float4) — T4 lacks cp.async hardware.
 * 2. Vectorized A loads: load float4 from global memory, scatter-transpose
 *    into shared memory (column-major layout for bank-conflict-free access).
 * 3. Added #pragma unroll on all inner loops.
 * 4. Added __restrict__ and --use_fast_math for compiler optimization.
 * 5. Proper double-buffering sync: 2x __syncthreads() per iteration to ensure
 *    (a) new data is ready before reading, (b) old data is not overwritten
 *    before all threads finish reading it.
 *
 * SMEM: 2*(BM*BK + BK*BN)*4 = 2*(128*16+16*128)*4 = 32KB (fits 2 blocks/SM)
 */

namespace {

template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *__restrict__ A,
                             const float *__restrict__ B, float *As, float *Bs,
                             int innerRowA, int innerColA, int innerRowB,
                             int innerColB) {
  // Load A tile with vectorized float4, then scatter-transpose into As[k][m]
  #pragma unroll
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  // Load B tile with vectorized float4 (already row-major, no transpose)
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
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults,
                const float *__restrict__ As, const float *__restrict__ Bs,
                const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  #pragma unroll
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    #pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      #pragma unroll
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    #pragma unroll
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      #pragma unroll
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    #pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      #pragma unroll
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        #pragma unroll
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          #pragma unroll
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace

/*
 * Double-buffered SGEMM kernel for SM 7.5+ (T4).
 *
 * Double-buffering with __syncthreads():
 *   Each iteration: load next tile → syncthreads → compute current tile → syncthreads
 *   The first syncthreads ensures the next tile's loads are visible.
 *   The second syncthreads ensures all threads are done reading the current
 *   tile before it gets overwritten in the next iteration.
 *
 *   Even though __syncthreads() is a hard barrier, the compiler still benefits
 *   because global loads are issued before the compute block, allowing memory
 *   latency to be partially hidden by the compute instructions that follow.
 */
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

  // Double-buffered shared memory
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
      N, K, A, B, &As[0], &Bs[0], innerRowA, innerColA, innerRowB, innerColB);
  __syncthreads(); // S1: first tile is ready

  // Main loop over K dimension (all tiles except the last)
  for (uint bkIdx = 0; bkIdx < K - BK; bkIdx += BK) {
    // Determine which buffer holds the current data and which gets the next
    int curBuf = (bkIdx / BK) & 1;      // 0, 1, 0, 1, ...
    int nextBuf = 1 - curBuf;

    // Issue loads for the NEXT tile into nextBuf
    // These global loads are pipelined — the SM can start fetching while
    // we're computing below.
    loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A + BK, B + BK * N, &As[nextBuf * BM * BK],
        &Bs[nextBuf * BK * BN], innerRowA, innerColA, innerRowB, innerColB);

    // Compute on the CURRENT tile (data is in curBuf, guaranteed ready by
    // the __syncthreads that preceded this iteration)
    processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regM, regN, threadResults, &As[curBuf * BM * BK],
        &Bs[curBuf * BK * BN], warpRow, warpCol, threadRowInWarp,
        threadColInWarp);

    A += BK;
    B += BK * N;

    // S2: ensures (a) all loads for nextBuf are committed to SMEM,
    //             (b) all reads from curBuf are complete (safe to overwrite)
    __syncthreads();
  }

  // Process the last tile
  {
    int lastBuf = ((K / BK - 1)) & 1;  // which buffer holds the last tile
    processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regM, regN, threadResults, &As[lastBuf * BM * BK],
        &Bs[lastBuf * BK * BN], warpRow, warpCol, threadRowInWarp,
        threadColInWarp);
  }

  // Write results back to global memory (vectorized float4 stores)
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
