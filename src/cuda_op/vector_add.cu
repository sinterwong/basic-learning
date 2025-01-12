#include "vector_add.cuh"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gtest/gtest.h>

namespace cuda_op {
template <typename T>
__global__ void vectorAddBasic(const T *A, const T *B, T *C, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

template <int BLOCK_SIZE, typename T>
__global__ void vectorAddShared(const T *A, const T *B, T *C, int N) {
  __shared__ T sharedA[BLOCK_SIZE];
  __shared__ T sharedB[BLOCK_SIZE];

  int tid = threadIdx.x;
  int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

  // load data to shared
  if (globalIdx < N) {
    sharedA[tid] = A[globalIdx];
    sharedB[tid] = B[globalIdx];
  }

  __syncthreads();

  if (globalIdx < N) {
    C[globalIdx] = sharedA[tid] + sharedB[tid];
  }
}

} // namespace cuda_op
