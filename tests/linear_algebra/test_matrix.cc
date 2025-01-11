#include "matrix.hpp"
#include <gtest/gtest.h>

using namespace linear_algebra;

static constexpr float EPSILON = 1e-6;

class MatricTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST(MatricTest, Size) {
  Matrix<int> m({{1, 2, 3}, {4, 5, 6}});
  ASSERT_EQ(m.size(), 6);
}

TEST(MatricTest, RowsAndCols) {
  Matrix<int> m({{1, 2, 3}, {4, 5, 6}});
  ASSERT_EQ(m.rows(), 2);
  ASSERT_EQ(m.cols(), 3);
}

TEST(MatricTest, Access) {
  Matrix<int> m({{1, 2, 3}, {4, 5, 6}});
  ASSERT_EQ(m[0][0], 1);
  ASSERT_EQ(m[0][1], 2);
  ASSERT_EQ(m[0][2], 3);
  ASSERT_EQ(m[1][0], 4);
  ASSERT_EQ(m[1][1], 5);
  ASSERT_EQ(m[1][2], 6);
}

TEST(MatricTest, Addition) {
  Matrix<int> m1({{1, 2, 3}, {4, 5, 6}});
  Matrix<int> m2({{7, 8, 9}, {10, 11, 12}});
  Matrix<int> expected({{8, 10, 12}, {14, 16, 18}});
  ASSERT_EQ(m1 + m2, expected);
}

TEST(MatricTest, Subtraction) {
  Matrix<int> m1({{1, 2, 3}, {4, 5, 6}});
  Matrix<int> m2({{7, 8, 9}, {10, 11, 12}});
  Matrix<int> expected({{-6, -6, -6}, {-6, -6, -6}});
  ASSERT_EQ(m1 - m2, expected);
}

TEST(MatricTest, Multiply) {
  Matrix<int> m1({{1, 2, 3}, {4, 5, 6}});
  Matrix<int> expected({{2, 4, 6}, {8, 10, 12}});
  ASSERT_EQ(m1 * 2, expected);
  ASSERT_EQ(2 * m1, expected);
}

TEST(MatricTest, Division) {
  Matrix<float> m1({{1, 2, 3}, {4, 5, 6}});
  Matrix<float> expected({{0.5, 1, 1.5}, {2, 2.5, 3}});
  ASSERT_EQ(m1 / 2, expected);
}

TEST(MatricTest, Equal) {
  Matrix<int> m1({{1, 2, 3}, {4, 5, 6}});
  Matrix<int> m2({{1, 2, 3}, {4, 5, 6}});
  ASSERT_TRUE(m1 == m2);

  Matrix<int> m3({{1, 2, 3}, {4, 5, 7}});
  ASSERT_FALSE(m1 == m3);
}

TEST(MatricTest, ZeroMat) {
  Matrix<int> expected({{0, 0, 0}, {0, 0, 0}});
  ASSERT_EQ(Matrix<int>::zero(2, 3), expected);
}

TEST(MatricTest, UnitMat) {
  Matrix<int> expected({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
  ASSERT_EQ(Matrix<int>::unit(3), expected);

  Matrix<int> A({{1, 2, 3}, {4, 5, 6}});
  ASSERT_EQ(A, A.dot(Matrix<int>::unit(3)));
}

TEST(MatricTest, DotVec) {
  Matrix<int> m({{1, 2, 3}, {4, 5, 6}});
  Vector<int> v({7, 8, 9});
  Vector<int> expected({50, 122});
  ASSERT_EQ(m.dot(v), expected);
}

TEST(MatricTest, DotMat) {
  Matrix<int> m1({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Matrix<int> m2({{7, 8}, {9, 10}, {11, 12}});
  Matrix<int> expected({{58, 64}, {139, 154}, {220, 244}});
  ASSERT_EQ(m1.dot(m2), expected);

  ASSERT_ANY_THROW(m2.dot(m1));
}

TEST(MatricTest, Transpose) {
  Matrix<int> m({{1, 2, 3}, {4, 5, 6}});
  Matrix<int> expected({{1, 4}, {2, 5}, {3, 6}});
  ASSERT_EQ(m.transpose(), expected);

  Matrix<int> A({{1, 2, 3}, {4, 5, 6}});
  Matrix<int> B({{1, 4}, {2, 5}, {3, 6}});
  ASSERT_EQ((A.dot(B)).transpose(), B.transpose().dot(A.transpose()));
}

TEST(MatricTest, Power) {
  Matrix<int> m({{1, 2}, {3, 4}});
  Matrix<int> expected({{7, 10}, {15, 22}});
  ASSERT_EQ(m.power(2), expected);
}
