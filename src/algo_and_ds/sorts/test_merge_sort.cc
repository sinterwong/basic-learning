/**
 * @file test_merge_sort.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-12-31
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <vector>

#include "merge_sort.hpp"
#include "select_sort.hpp"
#include "sort_helper.hpp"

using namespace algo_and_ds::sort;

int main(int argc, char const *argv[]) {

  int n = 50000;

  // 生成随机数组
  std::vector<int> arr_merge_rand, arr_selection_rand;
  arr_merge_rand.resize(n);

  // 生成随机数组
  generateRandomArray(arr_merge_rand.begin(), arr_merge_rand.end(), 0, 100);

  // 拷贝数组
  arr_selection_rand = arr_merge_rand; // copy

  testSort("merge sort(rand data): ", arr_merge_rand.begin(),
           arr_merge_rand.end(),
           [](auto first, auto last) { merge_sort(first, last); });

  testSort("Selection sort(rand data): ", arr_selection_rand.begin(),
           arr_selection_rand.end(),
           [](auto first, auto last) { select_sort(first, last); });

  return 0;
}