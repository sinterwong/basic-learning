#include <cstddef>
#include <iostream>
#include <array>
#include <cuda_runtime.h>
#include <numeric>

#define threadsPerBlock 256

__global__ void arraySum(const float* a, float *ret) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int block_id = blockIdx.x;
  int block_tid = threadIdx.x;
  __shared__ float sData[threadsPerBlock];  // 申请共享数据内存（每个block中的线程共享）
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

int main(int argc, char **argv) {

  // 数据数量需要符合2^n，不够需要补0
  constexpr int numElements = 5120;

  const int blockPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;  // block数量

  std::array<float, numElements> data;
  for (int i = 0; i < data.size(); i ++) {
    data[i] = i;
  }
  // for (auto i : data) {
  //   std::cout << i << ", ";
  // }
  // std::cout << std::endl;

  size_t size = numElements * sizeof(float);

  std::array<float, blockPerGrid> ret; // result on the host
  float *d_data = nullptr;  // data of device
  float *d_ret = nullptr; // result on the device
  cudaMalloc((void**)&d_data, size);
  cudaMalloc((void**)&d_ret, blockPerGrid * sizeof(float));

  cudaMemcpy(d_data, data.begin(), size, cudaMemcpyHostToDevice);
  
  arraySum<<<blockPerGrid, threadsPerBlock>>>(d_data, d_ret);
  cudaMemcpy(ret.begin(), d_ret, blockPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

  int result = std::accumulate(ret.begin(), ret.end(), 0);

  std::cout << "result: " <<  result << std::endl;
  cudaFree(d_data);
  cudaFree(d_ret);
  return 0;
}
