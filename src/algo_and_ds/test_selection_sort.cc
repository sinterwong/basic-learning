/**
 * @file test_selection_sort.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-12-30
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "select_sort.hpp"
#include "sort_helper.hpp"
#include <iostream>

using namespace algo_and_ds::sort;

int main(int argc, char *argv[]) {

  int n = 10;
  std::vector<int> arr;
  arr.resize(n);
  generateRandomArray(arr.begin(), arr.end(), 0, 100);

  testSort("Selection sort: ", arr.begin(), arr.end(),
           [](auto first, auto last) { select_sort(first, last); });

  printArray(arr.begin(), arr.end());

  return 0;
}
