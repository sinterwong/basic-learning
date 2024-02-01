/**
 * @file test_quick_sort.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-01-03
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "merge_sort.hpp"
#include "quick_sort.hpp"
#include "sort_helper.hpp"
#include <vector>

using namespace algo_and_ds::sort;

int main(int argc, char const *argv[]) {

  int n = 1000000;

  // 生成随机数组
  std::vector<int> arr_quick_rand, arr_merge_rand, arr_quick2_rand;
  arr_quick_rand.resize(n);
  generateRandomArray(arr_quick_rand.begin(), arr_quick_rand.end(), 0, n);
  arr_merge_rand = arr_quick_rand;  // copy
  arr_quick2_rand = arr_quick_rand; // copy

  testSort("quick sort 3way(rand data): ", arr_quick_rand.begin(),
           arr_quick_rand.end(),
           [](auto first, auto last) { quick_sort3way(first, last); });

  testSort("quick sort 2way(rand data): ", arr_quick2_rand.begin(),
           arr_quick2_rand.end(),
           [](auto first, auto last) { quick_sort2way(first, last); });

  testSort("merge sort(rand data): ", arr_merge_rand.begin(),
           arr_merge_rand.end(),
           [](auto first, auto last) { merge_sort(first, last); });

  // 生成近乎有序的数组
  std::vector<int> arr_quick_near, arr_merge_near, arr_quick2_near;
  arr_quick_near.resize(n);
  generateNearlyOrderedArray(arr_quick_near.begin(), arr_quick_near.end(), 100);
  arr_merge_near = arr_quick_near;  // copy
  arr_quick2_near = arr_quick_near; // copy

  testSort("quick sort 3way(near ordered data): ", arr_quick_near.begin(),
           arr_quick_near.end(),
           [](auto first, auto last) { quick_sort3way(first, last); });

  testSort("quick sort2(near ordered data): ", arr_quick2_near.begin(),
           arr_quick2_near.end(),
           [](auto first, auto last) { quick_sort2way(first, last); });

  testSort("merge sort(near ordered data): ", arr_merge_near.begin(),
           arr_merge_near.end(),
           [](auto first, auto last) { merge_sort(first, last); });

  // 生成有大量重复元素的数组
  std::vector<int> arr_quick_repeat, arr_merge_repeat, arr_quick2_repeat;
  arr_quick_repeat.resize(n);
  generateRandomArray(arr_quick_repeat.begin(), arr_quick_repeat.end(), 0, 10);
  arr_merge_repeat = arr_quick_repeat;  // copy
  arr_quick2_repeat = arr_quick_repeat; // copy

  testSort("quick sort 3way(repeat data): ", arr_quick_repeat.begin(),
           arr_quick_repeat.end(),
           [](auto first, auto last) { quick_sort3way(first, last); });

  testSort("quick sort2(repeat data): ", arr_quick2_repeat.begin(),
           arr_quick2_repeat.end(),
           [](auto first, auto last) { quick_sort2way(first, last); });

  testSort("merge sort(repeat data): ", arr_merge_repeat.begin(),
           arr_merge_repeat.end(),
           [](auto first, auto last) { merge_sort(first, last); });

  return 0;
}