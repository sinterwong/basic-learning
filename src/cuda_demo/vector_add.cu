#include "cuda_helper.hpp"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gtest/gtest.h>

namespace cuda_demo {
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

} // namespace cuda_demo

TEST(VectorAddTest, Normal) {
  const int N = 1 << 20;
  std::vector<int> h_A(N), h_B(N), h_C(N);
  for (int i = 0; i < N; ++i) {
    h_A[i] = i;
    h_B[i] = i * 2;
  }

  // alloc gpu memery
  cuda_demo::CudaArray<int> d_A(N), d_B(N), d_C(N);
  d_A.copyFromHost(h_A);
  d_B.copyFromHost(h_B);

  // set block and grid
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  // cuda_demo::vectorAddBasic<<<blocksPerGrid, threadsPerBlock>>>(
  //     d_A.get(), d_B.get(), d_C.get(), N);

  cuda_demo::vectorAddShared<threadsPerBlock>
      <<<blocksPerGrid, threadsPerBlock>>>(d_A.get(), d_B.get(), d_C.get(), N);

  cudaDeviceSynchronize();
  d_C.copyToHost(h_C);

  for (int i = 0; i < N; ++i) {
    ASSERT_FLOAT_EQ(h_C[i], h_A[i] + h_B[i]);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}