#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace cuda_op {
template <typename T>
__global__ void matrixMulBasic(const T *A, const T *B, T *C, int M, int K,
                               int N);
template <int TILE_SIZE, typename T>
__global__ void matrixMulShared(const T *A, const T *B, T *C, int M, int K,
                                int N);

template <typename T>
void matrixMul(const T *A, const T *B, T *C, int M, int K, int N);

} // namespace cuda_op
