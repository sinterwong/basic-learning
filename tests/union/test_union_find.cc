/**
 * @file test_union_find.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-01-31
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "union_find.hpp"

#include <chrono>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <ratio>

using namespace algo_and_ds::tree;

class UnionFindTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

template <typename UnionFind> void test_union_find(int const size) {
  UnionFind uf(size);

  auto startUnion = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < size; ++i) {
    int p = rand() % size;
    int q = rand() % size;
    uf.unionElements(p, q);
  }
  auto endUnion = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> unionDuration =
      endUnion - startUnion;

  auto startFind = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < size; ++i) {
    uf.find(rand() % size);
  }
  auto endFind = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> findDuration = endFind - startFind;

  std::cout << "Time taken for " << size
            << " union operations: " << unionDuration.count() / 1000 << " s"
            << std::endl;
  std::cout << "Time taken for " << size
            << " find operations: " << findDuration.count() / 1000 << " s"
            << std::endl;
}

TEST_F(UnionFindTest, Normal) {
  test_union_find<UnionFindTreeOpSize>(2000000);
  test_union_find<UnionFindTreeOpRank>(2000000);
}
