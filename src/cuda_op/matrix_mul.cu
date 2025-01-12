#include "cuda_helper.cuh"
#include "matrix_mul.cuh"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gtest/gtest.h>

namespace cuda_op {
template <typename T>
__global__ void matrixMulBasic(const T *A, const T *B, T *C, int M, int K,
                               int N) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < M && col < N) {
    T sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

template <int TILE_SIZE, typename T>
__global__ void matrixMulShared(const T *A, const T *B, T *C, int M, int K,
                                int N) {
  __shared__ T sharedA[TILE_SIZE][TILE_SIZE];
  __shared__ T sharedB[TILE_SIZE][TILE_SIZE];

  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  int ty = threadIdx.y;
  int tx = threadIdx.x;

  T sum = 0;
  for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
  }
}

template <typename T>
void matrixMul(const T *A, const T *B, T *C, int M, int K, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      T sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

} // namespace cuda_op
