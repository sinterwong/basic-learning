#ifndef __CUDA_OP_ARRAY_SUM_HPP_
#define __CUDA_OP_ARRAY_SUM_HPP_

#include <cuda_runtime.h>

namespace cuda_op {
__global__ void arraySum(const float *a, float *ret);

} // namespace cuda_op

#endif
