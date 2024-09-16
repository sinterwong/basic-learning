#ifndef __LINEAR_ALGEBRA_MATRIC_HPP_
#define __LINEAR_ALGEBRA_MATRIC_HPP_
#include "vector.hpp"
#include <ostream>
#include <utility>
#include <vector>
namespace linear_algebra {

template <typename T> class Matrix {
public:
  Matrix(std::vector<Vector<T>> values) : values(values) {}

  Matrix(std::vector<std::vector<T>> values)
      : values(std::vector<Vector<T>>(values.begin(), values.end())) {}

  std::pair<size_t, size_t> shape() const {
    if (values.empty()) {
      return {0, 0};
    }
    return {values.size(), values[0].size()};
  }

  size_t size() const { return shape().first * shape().second; }

  size_t rows() const { return shape().first; }

  size_t cols() const { return shape().second; }

  Vector<T> row_vector(int index) const { return values[index]; }

  Vector<T> col_vector(int index) const {
    std::vector<T> result;
    for (int i = 0; i < rows(); i++) {
      result.push_back(values[i][index]);
    }
    return Vector<T>(result);
  }

  T at(int row, int col) const { return values[row][col]; }

  Vector<T> dot(Vector<T> const &vec) const {
    if (cols() != vec.size()) {
      throw std::runtime_error("Matrix's column size must be equal to vector's"
                               "size");
    }
    std::vector<T> result;
    for (int i = 0; i < rows(); i++) {
      result.push_back(values[i].dot(vec));
    }
    return Vector<T>(result);
  }

  Matrix<T> dot(Matrix<T> const &other) const {
    if (cols() != other.rows()) {
      throw std::runtime_error("Matrix's column size must be equal to other"
                               "matrix's row size");
    }
    std::vector<Vector<T>> result;
    for (int i = 0; i < rows(); i++) {
      std::vector<T> row;
      for (int j = 0; j < other.cols(); j++) {
        row.push_back(values[i].dot(other.col_vector(j)));
      }
      result.push_back(Vector<T>(row));
    }
    return Matrix<T>(result);
  }

  Matrix<T> transpose() const {
    std::vector<Vector<T>> result;
    for (int i = 0; i < cols(); i++) {
      result.push_back(col_vector(i));
    }
    return Matrix<T>(result);
  }

  Matrix<T> power(int n) const {
    if (cols() != rows()) {
      throw std::runtime_error("Matrix must be square");
    }
    Matrix<T> result = *this;
    for (int i = 1; i < n; i++) {
      result = result.dot(*this);
    }
    return result;
  }

  Vector<T> &operator[](int index) { return values[index]; }

  const Vector<T> &operator[](int index) const { return values[index]; }

  bool operator==(Matrix<T> const &other) const {
    if (shape() != other.shape()) {
      return false;
    }
    for (int i = 0; i < shape().first; i++) {
      if (values[i] != other.values[i]) {
        return false;
      }
    }
    return true;
  }

  Matrix<T> operator+(Matrix<T> const &other) const {
    if (shape() != other.shape()) {
      throw std::runtime_error("Matrices must have the same shape");
    }
    std::vector<Vector<T>> result;
    for (int i = 0; i < shape().first; i++) {
      result.push_back(values[i] + other.values[i]);
    }
    return Matrix<T>(result);
  }

  Matrix<T> operator-(Matrix<T> const &other) const {
    if (shape() != other.shape()) {
      throw std::runtime_error("Matrices must have the same shape");
    }
    std::vector<Vector<T>> result;
    for (int i = 0; i < shape().first; i++) {
      result.push_back(values[i] - other.values[i]);
    }
    return Matrix<T>(result);
  }

  Matrix<T> operator*(T const &k) const {
    std::vector<Vector<T>> result;
    for (int i = 0; i < shape().first; i++) {
      result.push_back(values[i] * k);
    }
    return Matrix<T>(result);
  }

  Matrix<T> operator/(T const &k) const {
    std::vector<Vector<T>> result;
    for (int i = 0; i < shape().first; i++) {
      result.push_back(values[i] / k);
    }
    return Matrix<T>(result);
  }

  static Matrix<T> zero(size_t rows, size_t cols) {
    std::vector<Vector<T>> result;
    for (int i = 0; i < rows; i++) {
      result.push_back(Vector<T>::zero(cols));
    }
    return Matrix<T>(result);
  }

  static Matrix<T> unit(size_t size) {
    std::vector<Vector<T>> result;
    for (int i = 0; i < size; i++) {
      result.push_back(Vector<T>::zero(size));
      result[i][i] = 1;
    }
    return Matrix<T>(result);
  }

private:
  std::vector<Vector<T>> values;
};

template <typename T>
Matrix<T> operator*(T const &scalar, Matrix<T> const &mat) {
  return mat * scalar;
}

template <typename U>
std::ostream &operator<<(std::ostream &os, const Matrix<U> &mat) {
  os << "[";
  for (int i = 0; i < mat.rows(); i++) {
    os << mat[i];
    if (i < mat.rows() - 1) {
      os << std::endl;
    }
  }
  os << "]";
  return os;
}

} // namespace linear_algebra

#endif // __LINEAR_ALGEBRA_MATRIC_HPP_