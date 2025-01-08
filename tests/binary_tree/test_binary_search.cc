/**
 * @file test_binary_search.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "binary_search.hpp"
#include "quick_sort.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace algo_and_ds;

class BinarySearchTest : public ::testing::Test {
protected:
  void SetUp() override {
    int n = 10;
    arr.resize(n);
    arr = {2, 4, 5, 0, 1};
    // sort::generateRandomArray(arr.begin(), arr.end(), 0, n);
  }
  void TearDown() override {}

  std::vector<int> arr;
};

TEST_F(BinarySearchTest, Normal) {
  sort::quick_sort(arr.begin(), arr.end());
  auto iter = algo::binary_search(arr.begin(), arr.end(), 5);

  ASSERT_EQ(*iter, 5);
}

TEST_F(BinarySearchTest, Recursion) {
  sort::quick_sort(arr.begin(), arr.end());
  auto iter = algo::binary_search_recursion(arr.begin(), arr.end(), 5);

  ASSERT_EQ(*iter, 5);
}
