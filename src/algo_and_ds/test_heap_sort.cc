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
#include "merge_sort.hpp"
#include "quick_sort.hpp"
#include "sort_helper.hpp"
#include <vector>

using namespace algo_and_ds::sort;

int main(int argc, char **argv) {

  int n = 100000;

  // 生成随机数组
  std::vector<int> arr_quick_rand, arr_merge_rand, arr_heap_rand;
  arr_quick_rand.resize(n);
  generateRandomArray(arr_quick_rand.begin(), arr_quick_rand.end(), 0, n);
  arr_merge_rand = arr_quick_rand; // copy
  arr_heap_rand = arr_quick_rand;  // copy

  testSort("quick sort 3way(rand data): ", arr_quick_rand.begin(),
           arr_quick_rand.end(),
           [](auto first, auto last) { quick_sort3way(first, last); });

  testSort("merge sort(rand data): ", arr_merge_rand.begin(),
           arr_merge_rand.end(),
           [](auto first, auto last) { merge_sort(first, last); });

  testSort("heap1 sort(rand data): ", arr_heap_rand.begin(),
           arr_heap_rand.end(),
           [](auto first, auto last) { heap_sort1(first, last); });

  return 0;
}