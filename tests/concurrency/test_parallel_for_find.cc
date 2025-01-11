/**
 * @file test_parallel_for_find.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-05-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "parallel_for_find.hpp"
#include <algorithm>
#include <gtest/gtest.h>

TEST(ParallelForFindTest, Normal) {
  // 测试10万个随机元素中查找某值
  std::vector<int> v(100000);
  std::generate(v.begin(), v.end(), []() { return rand() % 1000; });
  auto it = my_concurrency::parallel_for_find(v.begin(), v.end(), 50);
  if (it == v.end()) {
    ASSERT_TRUE(false);
  }
  ASSERT_EQ(*it, 50);
}

TEST(ParallelForFindTest, Async) {
  std::vector<int> v(100000);
  std::generate(v.begin(), v.end(), []() { return rand() % 1000; });
  auto it = my_concurrency::parallel_for_find_async(v.begin(), v.end(), 50);
  if (it == v.end()) {
    ASSERT_TRUE(false);
  }
  ASSERT_EQ(*it, 50);
}
