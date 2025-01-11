/**
 * @file test_parallel_partial_sum.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-05-16
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "parallel_partial_sum.hpp"
#include <gtest/gtest.h>

TEST(ParallelPartialSumTest, Normal) {
  std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  my_concurrency::parallel_partial_sum(v.begin(), v.end());
  ASSERT_EQ(v[0], 1);
  ASSERT_EQ(v[1], 3);
  ASSERT_EQ(v[2], 6);
  ASSERT_EQ(v[3], 10);
  ASSERT_EQ(v[4], 15);
  ASSERT_EQ(v[5], 21);
  ASSERT_EQ(v[6], 28);
  ASSERT_EQ(v[7], 36);
  ASSERT_EQ(v[8], 45);
  ASSERT_EQ(v[9], 55);
}
