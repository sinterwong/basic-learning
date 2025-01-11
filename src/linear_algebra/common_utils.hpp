#ifndef __LINEAR_ALGEGRA_COMMON_UTILS_HPP_
#define __LINEAR_ALGEGRA_COMMON_UTILS_HPP_
#include <cstdlib>
#include <limits>
namespace linear_algebra {

template <typename T> inline bool isZero(T value) {
  const T epsilon = std::numeric_limits<T>::epsilon();
  return std::abs(value) < epsilon;
}
} // namespace linear_algebra

#endif