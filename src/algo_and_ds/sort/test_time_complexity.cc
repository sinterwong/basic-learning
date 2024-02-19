/**
 * @file test_time_complexity.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "common_utils.hpp"

#include "binary_search.hpp"
#include "merge_sort.hpp"
#include "quick_sort.hpp"
#include "select_sort.hpp"
#include "sort_helper.hpp"
#include <algorithm>

using namespace algo_and_ds::sort;
using namespace algo_and_ds::utils;

int main() {
  auto genRandomArr = [](auto first, auto last) {
    generateRandomArray(first, last, 0, 10000000);
  };

  auto genOrderArr = [](auto first, auto last) { generateRange(first, last); };

  testTimeByDataScaling(
      "Quick sort", 2, 10, 22, genRandomArr,
      [](auto first, auto last) { quick_sort3way(first, last); });
  std::cout << std::endl;

  testTimeByDataScaling("Merge sort", 2, 10, 22, genRandomArr,
                        [](auto first, auto last) { merge_sort(first, last); });
  std::cout << std::endl;

  testTimeByDataScaling(
      "Select sort", 2, 10, 15, genRandomArr,
      [](auto first, auto last) { select_sort(first, last); });
  std::cout << std::endl;

  testTimeByDataScaling(
      "Binary search", 2, 10, 28, genOrderArr,
      [](auto first, auto last) { binary_search(first, last, 100); });
}