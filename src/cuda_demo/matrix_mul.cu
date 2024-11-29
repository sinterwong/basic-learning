#include "cuda_helper.hpp"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gtest/gtest.h>

namespace cuda_demo {
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

} // namespace cuda_demo

TEST(MatrixMulTest, Normal) {
  int M = 1024;
  int K = 1024;
  int N = 1024;
  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C_cpu(M * N);
  std::vector<float> C_gpu(M * N);

  for (int i = 0; i < M * K; ++i) {
    A[i] = i % 10;
  }
  for (int i = 0; i < K * N; ++i) {
    B[i] = i % 10;
  }

  cuda_demo::matrixMul(A.data(), B.data(), C_cpu.data(), M, K, N);

  cuda_demo::CudaArray<float> cuda_A(M * K);
  cuda_demo::CudaArray<float> cuda_B(K * N);
  cuda_demo::CudaArray<float> cuda_C(M * N);

  cuda_A.copyFromHost(A);
  cuda_B.copyFromHost(B);

  dim3 blockDim(16, 16);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y);
  cuda_demo::matrixMulBasic<<<gridDim, blockDim>>>(cuda_A.get(), cuda_B.get(),
                                                   cuda_C.get(), M, K, N);
  cudaDeviceSynchronize();

  cuda_C.copyToHost(C_gpu);

  for (int i = 0; i < M * N; ++i) {
    ASSERT_NEAR(C_cpu[i], C_gpu[i], 1e-5);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}