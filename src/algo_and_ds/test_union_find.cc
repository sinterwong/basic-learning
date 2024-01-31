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
#include <iostream>
#include <ratio>

using namespace algo_and_ds::tree;

template <typename UnionFind> void test_union_find(int const size) {
  UnionFind uf(size);

  // 计时并操作
  auto startUnion = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < size; ++i) {
    int p = rand() % size;
    int q = rand() % size;
    uf.unionElements(p, q);
  }
  auto endUnion = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> unionDuration =
      endUnion - startUnion;

  // 计时查操作
  auto startFind = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < size; ++i) {
    uf.find(rand() % size);
  }
  auto endFind = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> findDuration = endFind - startFind;

  // 输出结果
  std::cout << "Time taken for " << size
            << " union operations: " << unionDuration.count() / 1000 << " s"
            << std::endl;
  std::cout << "Time taken for " << size
            << " find operations: " << findDuration.count() / 1000 << " s"
            << std::endl;
}

int main() {
  // test_union_find<UnionFindGroup>(30000);
  // test_union_find<UnionFindTreeBase>(30000);
  test_union_find<UnionFindTreeOpSize>(2000000);
  test_union_find<UnionFindTreeOpRank>(2000000);

  return 0;
}