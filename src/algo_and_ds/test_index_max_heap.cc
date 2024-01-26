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
#include <iostream>

using namespace algo_and_ds::tree;
int main(int argc, char **argv) {

  srand(time(nullptr));

  IndexMaxHeap<int> indexMaxHeap(500);

  for (int i = 0; i < 10; ++i) {
    indexMaxHeap.insert(i, rand() % 100);
  }

  indexMaxHeap.change(2, 1000);

  while (!indexMaxHeap.isEmpty()) {
    std::cout << indexMaxHeap.extractMax() << ", ";
  }
  std::cout << std::endl;

  return 0;
}