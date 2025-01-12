#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace cuda_op {

class CudaError : public std::runtime_error {
public:
  CudaError(const char *msg, const char *func, const char *file, int line)
      : std::runtime_error(std::string(msg) + " at " + func + " (" + file +
                           ":" + std::to_string(line) + ")") {}
};

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
inline void check_cuda_error(cudaError_t err, const char *func,
                             const char *file, const int line) {
  if (err != cudaSuccess) {
    throw CudaError(cudaGetErrorString(err), func, file, line);
  }
}

template <typename T> class CudaArray {
private:
  T *ptr_ = nullptr;
  size_t size_ = 0;

public:
  CudaArray(size_t size) : size_(size) {
    CHECK_CUDA_ERROR(cudaMalloc(&ptr_, size_ * sizeof(T)));
  }

  ~CudaArray() {
    if (ptr_) {
      cudaFree(ptr_);
    }
  }

  CudaArray(const CudaArray &) = delete;
  CudaArray &operator=(const CudaArray &) = delete;

  CudaArray(CudaArray &&other) noexcept
      : ptr_(other.ptr_), size_(other.size_t) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  void copyFromHost(const std::vector<T> &host_data) {
    CHECK_CUDA_ERROR(cudaMemcpy(ptr_, host_data.data(), size_ * sizeof(T),
                                cudaMemcpyHostToDevice));
  }

  void copyToHost(std::vector<T> &host_data) {
    CHECK_CUDA_ERROR(cudaMemcpy(host_data.data(), ptr_, size_ * sizeof(T),
                                cudaMemcpyDeviceToHost));
  }

  T *get() { return ptr_; }
  const T *get() const { return ptr_; }
  size_t size() const { return size_; }
};

} // namespace cuda_op