/**
 * @file select_sort.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * 选择排序，O(n^2)排序算法，第一轮遍历维护整个数组的交换，第二轮遍历查找获取最小元素的位置
 * @version 0.1
 * @date 2022-09-10
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __AADS_SELECT_SORT_HPP_
#define __AADS_SELECT_SORT_HPP_
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>

namespace algo_and_ds::sort {
template <typename Container> void select_sort(Container &arr) {
  for (int i = 0; i < arr.size(); ++i) {
    // 剩余元素中最小的元素索引
    int minindex = i; // 对于T类型，不知如何表示最大值
    // j从i开始（i以前的已经有序）
    for (int j = i; j < arr.size(); ++j) {
      if (arr[j] < arr[minindex]) {
        minindex = j;
      }
    }
    if (i != minindex) {
      std::swap(arr[i], arr[minindex]);
    }
  }
}

template <typename Iter> void select_sort(Iter begin, Iter end) {
  for (auto i = begin; i != end; ++i) {
    auto minIndex = i;
    for (auto j = i; j != end; ++j) {
      if (*j < *minIndex) {
        minIndex = j;
      }
    }
    if (i != minIndex) {
      std::swap(*i, *minIndex);
    }
  }
}

template <typename Iter> void select_sort_recursive(Iter begin, Iter end) {
  if (begin == end) {
    return;
  }
  auto minIndex = begin;
  for (auto i = begin; i != end; ++i) {
    if (*i < *minIndex) {
      minIndex = i;
    }
  }
  if (begin != minIndex) {
    std::swap(*begin, *minIndex);
  }
  select_sort_recursive(begin + 1, end);
}
} // namespace algo_and_ds::sort

#endif