#include "vector.hpp"
#include <gtest/gtest.h>

using namespace linear_algebra;

static constexpr float EPSILON = 1e-6;

class VectorTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST(VectorTest, Size) {
  Vector<int> v({1, 2, 3});
  ASSERT_EQ(v.size(), 3);
}

TEST(VectorTest, Access) {
  Vector<int> v({1, 2, 3});
  ASSERT_EQ(v[0], 1);
  ASSERT_EQ(v[1], 2);
  ASSERT_EQ(v[2], 3);
}

TEST(VectorTest, Addition) {
  Vector<int> v1({1, 2, 3});
  Vector<int> v2({4, 5, 6});
  Vector<int> expected({5, 7, 9});
  ASSERT_EQ(v1 + v2, expected);
}

TEST(VectorTest, Subtraction) {
  Vector<int> v1({1, 2, 3});
  Vector<int> v2({4, 5, 6});
  Vector<int> expected({-3, -3, -3});
  ASSERT_EQ(v1 - v2, expected);
}

TEST(VectorTest, Multiply) {
  Vector<int> v1({1, 2, 3});
  Vector<int> expected({2, 4, 6});
  ASSERT_EQ(v1 * 2, expected);
  ASSERT_EQ(2 * v1, expected);
}

TEST(VectorTest, Positive) {
  Vector<int> v1({1, 2, 3});
  Vector<int> expected({1, 2, 3});
  ASSERT_EQ(+v1, expected);
}

TEST(VectorTest, Negative) {
  Vector<int> v1({1, 2, 3});
  Vector<int> expected({-1, -2, -3});
  ASSERT_EQ(-v1, expected);
}

TEST(VectorTest, ZeroVec) {
  Vector<int> expected({0, 0, 0});
  ASSERT_EQ(Vector<int>::zero(3), expected);
}

TEST(VectorTest, Norm) {
  Vector<float> vec1({3, 4});
  ASSERT_NEAR(vec1.norm(), 5.0, EPSILON);
  ASSERT_NEAR(vec1.normlize().norm(), 1.0, EPSILON);

  auto vec2 = Vector<float>::zero(2);
  ASSERT_NEAR(vec2.norm(), 0.0, EPSILON);
  ASSERT_THROW(vec2.normlize(), std::runtime_error);
}

TEST(VectorTest, Dot) {
  // acute angle
  Vector<int> vec1({1, 2});
  Vector<int> vec2({3, 4});
  ASSERT_EQ(vec1.dot(vec2), 11);

  // obtuse angle
  Vector<int> vec3({1, 2});
  Vector<int> vec4({-3, -4});
  ASSERT_EQ(vec3.dot(vec4), -11);

  // orthogonal
  Vector<int> vec5({3, 0});
  Vector<int> vec6({0, 3});
  ASSERT_EQ(vec5.dot(vec6), 0);

  // calc cosθ
  Vector<float> vec7({3, 0});
  Vector<float> vec8({0, 3});
  ASSERT_NEAR(vec7.dot(vec8) / (vec7.norm() * vec8.norm()), 0.0, EPSILON);

  // calc coordinates of the projection point
  Vector<float> vec9({3, 4});
  Vector<float> vec10({2, 8});
  auto projection = vec9.dot(vec10) / vec10.norm() * vec10.normlize();
  std::cout << projection << std::endl;
}

TEST(VectorTest, Equal) {
  Vector<int> v1({1, 2, 3});
  Vector<int> v2({1, 2, 3});
  ASSERT_TRUE(v1 == v2);

  Vector<int> v3({1, 2, 4});
  ASSERT_FALSE(v1 == v3);
}

TEST(VectorTest, Unequal) {
  Vector<int> v1({1, 2, 3});
  Vector<int> v2({1, 2, 4});
  ASSERT_NE(v1, v2);

  Vector<int> v3({1, 2, 3});
  ASSERT_EQ(v1, v3);
}

TEST(VectorTest, Iterator) {
  Vector<int> v({1, 2, 3});
  auto it = v.begin();
  ASSERT_EQ(*it, 1);
  ++it;
  ASSERT_EQ(*it, 2);
  ++it;
  ASSERT_EQ(*it, 3);
  ++it;
  ASSERT_EQ(it, v.end());
}

TEST(VectorTest, ConstIterator) {
  Vector<int> v({1, 2, 3});
  auto it = v.begin();
  ASSERT_EQ(*it, 1);
  ++it;
  ASSERT_EQ(*it, 2);
  ++it;
  ASSERT_EQ(*it, 3);
  ++it;
  ASSERT_EQ(it, v.end());
}

TEST(VectorTest, Stream) {
  Vector<int> v({1, 2, 3});
  std::stringstream ss;
  ss << v;
  ASSERT_EQ(ss.str(), "[1, 2, 3]");
}
