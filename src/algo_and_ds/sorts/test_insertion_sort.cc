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

#include "insert_sort.hpp"
#include "select_sort.hpp"
#include "sort_helper.hpp"
#include <iostream>

using namespace algo_and_ds::sort;

int main(int argc, char *argv[]) {

  int n = 10000;

  // 生成随机数组
  std::vector<int> arr_insertion_rand, arr_selection_rand;
  std::vector<int> arr_insertion_nearly_ordered, arr_selection_nearly_ordered;
  arr_insertion_rand.resize(n);
  arr_insertion_nearly_ordered.resize(n);

  // 生成随机数组
  generateRandomArray(arr_insertion_rand.begin(), arr_insertion_rand.end(), 0,
                      100);

  // 生成近乎有序的数组
  generateNearlyOrderedArray(arr_insertion_nearly_ordered.begin(),
                             arr_insertion_nearly_ordered.end(), 10);

  // 拷贝数组
  arr_selection_rand = arr_insertion_rand; // copy
  arr_selection_nearly_ordered = arr_insertion_nearly_ordered;

  testSort("Insertion sort(rand data): ", arr_insertion_rand.begin(),
           arr_insertion_rand.end(),
           [](auto first, auto last) { insert_sort_recursive(first, last); });

  testSort("Selection sort(rand data): ", arr_selection_rand.begin(),
           arr_selection_rand.end(),
           [](auto first, auto last) { select_sort_recursive(first, last); });

  testSort("Insertion sort(nearly ordered data): ",
           arr_insertion_nearly_ordered.begin(),
           arr_insertion_nearly_ordered.end(),
           [](auto first, auto last) { insert_sort(first, last); });

  testSort("Selection sort(nearly ordered data): ",
           arr_selection_nearly_ordered.begin(),
           arr_selection_nearly_ordered.end(),
           [](auto first, auto last) { select_sort(first, last); });

  return 0;
}
