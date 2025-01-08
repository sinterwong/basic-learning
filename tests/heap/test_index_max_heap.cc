/**
 * @file test_index_max_heap.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-01-26
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "index_max_heap.hpp"
#include <cstdlib>
#include <ctime>
#include <gtest/gtest.h>
#include <iostream>
using namespace algo_and_ds::tree;

class IndexMaxHeapTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(IndexMaxHeapTest, Normal) {
  IndexMaxHeap<int> indexMaxHeap(100);
  for (int i = 0; i < 100; i++) {
    indexMaxHeap.insert(i, i);
  }
  for (int i = 99; i >= 0; i--) {
    ASSERT_EQ(indexMaxHeap.extractMax(), i);
  }
}