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

#include "insert_sort.hpp"
#include "merge_sort.hpp"
#include "sort_helper.hpp"

using namespace algo_and_ds::sort;

int main(int argc, char const *argv[]) {

  int n = 30000;

  // 生成随机数组
  std::vector<int> arr_merge_rand, arr_merge_bu_rand, arr_insert_rand;
  arr_merge_rand.resize(n);

  // 生成随机数组
  generateRandomArray(arr_merge_rand.begin(), arr_merge_rand.end(), 0, 100);

  // 拷贝数组
  arr_insert_rand = arr_merge_rand;   // copy
  arr_merge_bu_rand = arr_merge_rand; // copy

  testSort("merge sort(rand data): ", arr_merge_rand.begin(),
           arr_merge_rand.end(),
           [](auto first, auto last) { merge_sort(first, last); });

  testSort("merge sort bottom up(rand data): ", arr_merge_bu_rand.begin(),
           arr_merge_bu_rand.end(),
           [](auto first, auto last) { merge_sort_bu(first, last); });

  testSort("Insertion sort(rand data): ", arr_insert_rand.begin(),
           arr_insert_rand.end(),
           [](auto first, auto last) { insert_sort(first, last); });

  // 生成近乎有序的数组
  std::vector<int> arr_merge_nearly_ordered, arr_merge_bu_nearly_ordered,
      arr_insert_nearly_ordered;
  arr_merge_nearly_ordered.resize(n);

  generateNearlyOrderedArray(arr_merge_nearly_ordered.begin(),
                             arr_merge_nearly_ordered.end(), 10);

  // 拷贝数组
  arr_insert_nearly_ordered = arr_merge_nearly_ordered;   // copy
  arr_merge_bu_nearly_ordered = arr_merge_nearly_ordered; // copy

  testSort("merge sort(nearly ordered data): ",
           arr_merge_nearly_ordered.begin(), arr_merge_nearly_ordered.end(),
           [](auto first, auto last) { merge_sort(first, last); });

  testSort("merge sort bottom up(nearly ordered data): ",
           arr_merge_bu_nearly_ordered.begin(),
           arr_merge_bu_nearly_ordered.end(),
           [](auto first, auto last) { merge_sort_bu(first, last); });

  testSort("Insertion sort(nearly ordered data): ",
           arr_insert_nearly_ordered.begin(), arr_insert_nearly_ordered.end(),
           [](auto first, auto last) { insert_sort(first, last); });

  return 0;
}