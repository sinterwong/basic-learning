/**
 * @file test_heap_sort.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-01-26
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "heap_sort.hpp"
#include "sort_helper.hpp"
#include <vector>

#include <gtest/gtest.h>

using namespace algo_and_ds::sort;

class HeapSortTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
  int n = 100000;
};

TEST_F(HeapSortTest, Normal) {
  std::vector<int> arrHeapRand;
  arrHeapRand.resize(n);
  generateRandomArray(arrHeapRand.begin(), arrHeapRand.end(), 0, n);

  testSort("heap1 sort(rand data): ", arrHeapRand.begin(), arrHeapRand.end(),
           [](auto first, auto last) { heap_sort1(first, last); });
}
