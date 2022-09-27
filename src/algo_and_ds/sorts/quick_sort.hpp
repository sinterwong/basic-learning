/**
 * @file quick_sort.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * 快速排序，O(nlogn)排序算法，每次找到一个基点，对基点进行partition操作（将基点左侧都小于它，右侧都大于它）
 然后左右子树分别在递归下去。
 * @version 0.1
 * @date 2022-09-10
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __AADS_QUICK_SORT_HPP_
#define __AADS_QUICK_SORT_HPP_
#include "insert_sort.hpp"
#include <algorithm>
#include <iterator>
#include <memory>
#include <tuple>

namespace algo_and_ds {
namespace sort {

template <typename Container>
int __partition(Container &arr, int l, int r) {
  using valType =
      typename std::iterator_traits<typename Container::iterator>::value_type;
  // 取第一个元素作为基点
  valType v = arr[l];
  
  // j用来维护arr[l..p-1], i维护arr[p+1....r)
  int j = l;
  for (int i = l + 1; i <= r; i ++) {
    if (arr[i] < v) {  // 需要放到v的右侧，然后更新j
      std::swap(arr[i], arr[++j]);
    }
  }
  std::swap(arr[j], arr[l]);  // 此时j的位置指向最后一个小于v的元素
  return j;


}

template <typename Container> void __quickSort(Container &arr, int l, int r) {
  if (l >= r) {
    return;
  }
  int pid = __partition(arr, l, r);
  __quickSort(arr, l, pid-1);
  __quickSort(arr, pid+1, r);
}

template <typename Container> void quick_sort(Container &arr) {
  __quickSort(arr, 0, arr.size());
}

} // namespace sort
} // namespace algo_and_ds

#endif