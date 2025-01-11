#include "linear_systemp.hpp"
#include <gtest/gtest.h>

using namespace linear_algebra;

static constexpr float EPSILON = 1e-6;

class LinearSystemTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST(LinearSystemTest, GaussJordanElimination) {
  Matrix<float> A({{1, 2, 4}, {3, 7, 2}, {2, 3, 3}});
  Vector<float> b({7, -11, 1});
  LinearSystem<float> ls(A, b);
  ls.gaussJordanElimination();
  ASSERT_NEAR(ls.getAb()[0][3], -1.0, EPSILON);
  ASSERT_NEAR(ls.getAb()[1][3], -2.0, EPSILON);
  ASSERT_NEAR(ls.getAb()[2][3], 3.0, EPSILON);
}

TEST(LinearSystemTest, GaussJordanEliminationNormal) {
  Matrix<float> A({{1, -1, 2, 0, 3},
                   {-1, 1, 0, 2, -5},
                   {1, -1, 4, 2, 4},
                   {-2, 2, -5, -1, -3}});
  Vector<float> b({1, 5, 13, -1});
  LinearSystem<float> ls(A, b);
  ASSERT_TRUE(ls.gaussJordanElimination());
  ASSERT_NEAR(ls.getAb()[0][5], -15, EPSILON);
  ASSERT_NEAR(ls.getAb()[1][5], 5, EPSILON);
  ASSERT_NEAR(ls.getAb()[2][5], 2, EPSILON);
}

TEST(LinearSystemTest, GaussJordanEliminationNoSolution) {
  Matrix<float> A({{2, 2}, {2, 1}, {1, 2}});
  Vector<float> b({3, 2.5, 7});
  LinearSystem<float> ls(A, b);
  ASSERT_FALSE(ls.gaussJordanElimination());
  ASSERT_NEAR(ls.getAb()[0][2], -4, EPSILON);
  ASSERT_NEAR(ls.getAb()[1][2], 5.5, EPSILON);
  ASSERT_NEAR(ls.getAb()[2][2], 5, EPSILON);
}

TEST(LinearSystemTest, Invert) {
  Matrix<float> A({{1, 2}, {3, 4}});
  auto invA = invert(A);
  ASSERT_NEAR(invA[0][0], -2, EPSILON);
  ASSERT_NEAR(invA[0][1], 1, EPSILON);
  ASSERT_NEAR(invA[1][0], 1.5, EPSILON);
  ASSERT_NEAR(invA[1][1], -0.5, EPSILON);
}
