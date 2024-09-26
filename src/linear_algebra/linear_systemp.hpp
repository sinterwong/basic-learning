#include "common_utils.hpp"
#include "matrix.hpp"
#include "vector.hpp"
#include <cstddef>

#ifndef __LINEAR_ALGEBRA_LINEAR_SYSTEM_HPP_
#define __LINEAR_ALGEBRA_LINEAR_SYSTEM_HPP_

namespace linear_algebra {

template <typename T> class LinearSystem {
public:
  LinearSystem(Matrix<T> const &A, Vector<T> const &b) {
    if (A.rows() != b.size()) {
      throw std::runtime_error("Matrix's row size must be equal to vector's"
                               "size");
    }
    row = A.rows();
    col = A.cols();

    Ab = A;
    Ab.add_col(b);
  }

  LinearSystem(Matrix<T> const &A, Matrix<T> const &b) {
    if (A.rows() != b.rows()) {
      throw std::runtime_error("Matrix's row size must be equal to other"
                               "matrix's row size");
    }
    row = A.rows();
    col = A.cols();

    Ab = A;
    Ab.add_col(b);
  }

  bool gaussJordanElimination() {
    forward();
    backward();

    for (int i = pivots.size(); i < row; ++i) {
      for (int j = col; j < Ab.cols(); ++j) {
        if (!isZero(Ab[i][j])) {
          return false;
        }
      }
    }
    return true;
  }

  Matrix<T> getAb() { return Ab; }

  Matrix<T> getA() {
    Matrix<T> A;
    for (int i = 0; i < col; ++i) {
      A.add_col(Ab.col_vector(i));
    }
    return A;
  }

  Matrix<T> getb() {
    Matrix<T> b;
    for (int i = col; i < Ab.cols(); ++i) {
      b.add_col(Ab.col_vector(i));
    }
    return b;
  }

  bool existsZeroRow() {
    for (int i = 0; i < row; ++i) {
      bool ret = true;
      for (int j = 0; j < col; ++j) {
        if (!isZero(Ab[i][j])) {
          ret = false;
          break;
        }
      }
      if (ret) {
        return true;
      }
    }
    return false;
  }

private:
  void forward() {
    int r = 0;
    int c = 0;
    while (r < row && c < col) {
      auto maxRow = calcMaxRow(r, c);
      if (maxRow != r) {
        std::swap(Ab[r], Ab[maxRow]);
      }
      if (std::abs(Ab[r][c]) < 1e-6) {
        c++;
        continue;
      }

      // make pivot to 1
      Ab[r] = Ab[r] / Ab[r][c];

      // make non-pivot to 0
      for (int j = r + 1; j < row; ++j) {
        Ab[j] = Ab[j] - (Ab[j][c] * Ab[r]);
      }
      pivots.push_back(c);
      r++;
    }
  }

  void backward() {
    size_t n = pivots.size();
    for (int i = n - 1; i > 0; --i) {
      int c = pivots[i];
      for (int j = i - 1; j >= 0; --j) {
        Ab[j] = Ab[j] - (Ab[j][c] * Ab[i]);
      }
    }
  }

  size_t calcMaxRow(size_t index_i, size_t index_j) {
    auto maxRow = index_i;
    for (int i = index_i + 1; i < row; ++i) {
      if (Ab[i][index_j] > Ab[maxRow][index_j]) {
        maxRow = i;
      }
    }
    return maxRow;
  }

private:
  Matrix<T> Ab;
  size_t row;
  size_t col;
  std::vector<size_t> pivots;
};

template <typename T> Matrix<T> invert(Matrix<T> const &A) {
  if (A.rows() != A.cols()) {
    throw std::runtime_error("Matrix must be square");
  }
  auto unit = Matrix<T>::unit(A.rows());

  LinearSystem<T> ls(A, unit);
  ls.gaussJordanElimination();
  if (ls.existsZeroRow()) {
    return Matrix<T>();
  }
  return ls.getb();
}
} // namespace linear_algebra

#endif // __LINEAR_ALGEBRA_LINEAR_SYSTEM_HPP_