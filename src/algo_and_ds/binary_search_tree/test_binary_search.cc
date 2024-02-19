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
#include <vector>

using namespace algo_and_ds;

int main() {

  int n = 100000;
  std::vector<int> arr;
  arr.resize(n);
  sort::generateRandomArray(arr.begin(), arr.end(), 0, 100);

  sort::quick_sort(arr.begin(), arr.end());

  auto iter = algo::binary_search(arr.begin(), arr.end(), 10);

  if (iter == arr.end()) {
    std::cout << "Not found!" << std::endl;
    return -1;
  }

  std::cout << "Ret: " << *iter << std::endl;
  return 0;
}