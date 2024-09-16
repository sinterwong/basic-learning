/**
 * @file vector.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-09-16
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __LINEAR_ALGEBRA_VECTOR_HPP_
#define __LINEAR_ALGEBRA_VECTOR_HPP_

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <ostream>
#include <vector>

namespace linear_algebra {

template <typename T> class Vector {
public:
  Vector(std::vector<T> values) : values(values) {}

  size_t size() const { return values.size(); }

  T &operator[](int index) { return values[index]; }

  const T &operator[](int index) const { return values[index]; }

  Vector<T> operator+(Vector<T> const &other) const {
    if (values.size() != other.size()) {
      throw std::runtime_error("Vectors must have the same size");
    }
    std::vector<T> result;
    std::transform(values.begin(), values.end(), other.begin(),
                   std::back_inserter(result),
                   [](T const &v1, T const &v2) { return v1 + v2; });
    return Vector<T>(result);
  }

  Vector<T> operator-(Vector<T> const &other) const {
    if (values.size() != other.size()) {
      throw std::runtime_error("Vectors must have the same size");
    }
    std::vector<T> result;
    std::transform(values.begin(), values.end(), other.begin(),
                   std::back_inserter(result),
                   [](T const &v1, T const &v2) { return v1 - v2; });
    return Vector<T>(result);
  }

  Vector<T> operator*(T const &k) const {
    std::vector<T> result;
    std::transform(values.begin(), values.end(), std::back_inserter(result),
                   [&k](T const &val) { return k * val; });
    return Vector<T>(result);
  }

  Vector<T> operator/(T const &k) const {
    std::vector<T> result;
    std::transform(values.begin(), values.end(), std::back_inserter(result),
                   [&k](T const &val) { return val / k; });
    return Vector<T>(result);
  }

  Vector<T> operator+() const { return *this; }

  Vector<T> operator-() const {
    std::vector<T> result;
    std::transform(values.begin(), values.end(), std::back_inserter(result),
                   std::negate<>());
    return Vector<T>(result);
  }

  bool operator==(Vector<T> const &other) const {
    if (values.size() != other.size()) {
      throw std::runtime_error("Vectors must have the same size");
    }
    return std::equal(values.begin(), values.end(), other.values.begin());
  }

  T norm() const {
    return std::sqrt(
        std::inner_product(values.begin(), values.end(), values.begin(), 0.0));
  }

  Vector<T> normlize() const {
    if (norm() == 0)
      throw std::runtime_error("Vector's norm is zero");
    return *this / norm();
  }

  T dot(Vector<T> const &other) const {
    return std::inner_product(values.begin(), values.end(),
                              other.values.begin(), 0.0);
  }

  typename std::vector<T>::iterator begin() { return values.begin(); }

  typename std::vector<T>::iterator end() { return values.end(); }

  typename std::vector<T>::const_iterator begin() const {
    return values.begin();
  }

  typename std::vector<T>::const_iterator end() const { return values.end(); }

  static Vector<T> zero(size_t size) {
    return Vector<T>(std::vector<T>(size, 0));
  }

private:
  std::vector<T> values;
};

template <typename U>
std::ostream &operator<<(std::ostream &os, const Vector<U> &vec) {
  os << "[";
  for (auto i = 0; i < vec.size(); i++) {
    os << vec[i];
    if (i < vec.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

template <typename T>
Vector<T> operator*(T const &scalar, Vector<T> const &vec) {
  return vec * scalar;
}

} // namespace linear_algebra

#endif // __LINEAR_ALGEBRA_VECTOR_HPP_