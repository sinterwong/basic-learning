#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gtest/gtest.h>

namespace cuda_op {
template <typename T>
__global__ void vectorAddBasic(const T *A, const T *B, T *C, int N);

template <int BLOCK_SIZE, typename T>
__global__ void vectorAddShared(const T *A, const T *B, T *C, int N);

} // namespace cuda_op