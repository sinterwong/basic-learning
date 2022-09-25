/**
 * @file insert_sort.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * 插入排序，O(n^2)排序算法，第一轮遍历过每一个元素，每一个元素再分别向前比较，直到找到了自己的位置
 * @version 0.1
 * @date 2022-09-10
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __AADS_INSERT_SORT_HPP_
#define __AADS_INSERT_SORT_HPP_
#include <iterator>
#include <memory>

namespace algo_and_ds {
namespace sort {

template <typename Container> void insert_sort(Container &arr, int l, int r) {
  using valType =
      typename std::iterator_traits<typename Container::iterator>::value_type;
  for (int i = l + 1; i <= r; i++) {
    valType temp = arr[i];
    int j;
    for (j = i; j > l && arr[j - 1] > temp; j--) {
      arr[j] = arr[j - 1];
    }
    arr[j] = temp;
  }
}

template <typename Container> void insert_sort(Container &arr) {
  using valType =
      typename std::iterator_traits<typename Container::iterator>::value_type;
  for (int i = 1; i < arr.size(); ++i) {
    valType temp = arr[i];
    int j;
    for (j = i; j > 0 && arr[j - 1] > temp; j--) {
      arr[j] = arr[j - 1];
    }
    arr[j] = temp;
  }
}

} // namespace sort
} // namespace algo_and_ds

#endif