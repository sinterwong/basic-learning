#include "array_sum.cuh"
#include "common_macro.hpp"
#include <cuda_runtime.h>

namespace cuda_op {
__global__ void arraySum(const float *a, float *ret) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int block_id = blockIdx.x;
  int block_tid = threadIdx.x;
  // 申请共享数据内存（每个block中的线程共享）
  __shared__ float sData[threadsPerBlock];
  sData[block_tid] = a[tid];
  __syncthreads();
  for (int i = threadsPerBlock / 2; i > 0; i /= 2) {
    if (block_tid < i) {
      sData[block_tid] = sData[block_tid] + sData[block_tid + i];
    }
    __syncthreads();
  }
  if (block_tid == 0) {
    ret[block_id] = sData[0];
  }
}

} // namespace cuda_op
