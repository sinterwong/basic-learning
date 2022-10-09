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

template <typename Container> int __partition(Container &arr, int l, int r) {
  using valType =
      typename std::iterator_traits<typename Container::iterator>::value_type;

  std::swap(arr[l],
            arr[rand() % (r - l + 1) + l]); // 交换一下第一个位置和随机元素

  valType v = arr[l]; // 取第一个元素作为基点

  // j用来维护arr[l..p-1], i维护arr[p+1....r)
  int j = l;
  for (int i = l + 1; i <= r; i++) {
    if (arr[i] < v) { // 需要放到v的右侧，然后更新j
      std::swap(arr[i], arr[++j]);
    }
  }
  std::swap(arr[j], arr[l]); // 此时j的位置指向最后一个小于v的元素
  return j;
}

template <typename Container> void __quickSort(Container &arr, int l, int r) {
  if (l >= r) {
    return;
  }
  int pid = __partition(arr, l, r);
  __quickSort(arr, l, pid - 1);
  __quickSort(arr, pid + 1, r);
}

template <typename Container> void quick_sort(Container &arr) {
  __quickSort(arr, 0, arr.size());
}

template <typename Container>
int __partition2way(Container &arr, int l, int r) {
  /**
   * @brief
   * 之前的快排对于重复元素极多的情况将会有非常大的概率两边不平衡，导致最坏回退化成O(n^2)
   * 双路快排通过尽可能均分两路，来实现一个期望平衡的左右子树。
   * 左右各自维护一个索引，左边向右遍历查找直到找到>v的元素交换，右边向左查找直到找到<v的进行交换，
   * 如此往复直到维护的两个索引重合
   */
  using valType =
      typename std::iterator_traits<typename Container::iterator>::value_type;

  std::swap(arr[l],
            arr[rand() % (r - l + 1) + l]); // 交换一下第一个位置和随机元素

  valType v = arr[l]; // 取第一个元素作为基点

  int i = l + 1; // arr[l+1...i)
  int j = r;     // arr(j...r]

  while (true) {
    while (i <= r && arr[i] < v)
      i++; // 小于v时循环继续直到 >=v时停止
    while (j >= l + 1 && arr[j] > v)
      j--;
    if (i >= j)
      break;
    std::swap(arr[j], arr[i]);
    i++;
    j--;
  }
  std::swap(arr[l], arr[j]);
  return j;
}

template <typename Container>
void __quickSort2way(Container &arr, int l, int r) {
  if (l >= r) {
    return;
  }
  int pid = __partition2way(arr, l, r);
  __quickSort2way(arr, l, pid - 1);
  __quickSort2way(arr, pid + 1, r);
}

template <typename Container> void quick_sort_2way(Container &arr) {
  __quickSort2way(arr, 0, arr.size());
}

template <typename Container>
void __quickSort3way(Container &arr, int l, int r) {
  if (l >= r) {
    return;
  }
  using valType =
      typename std::iterator_traits<typename Container::iterator>::value_type;
  // 分别维护小于v的部分，等于v的部分和大于v的部分，因此两个点就可以将数组划分成三段，所以需要维护两个索引
  // 初始化两个索引
  std::swap(arr[l], arr[rand() % (r - l + 1) + l]);
  valType v = arr[l];
  int lt = l;     // [l....lt]
  int gt = r + 1; // [gt....r]
  int i = l + 1;  // 前进索引
  while (i < gt) {
    if (arr[i] < v) {
      std::swap(arr[++lt], arr[i++]);
    } else if (arr[i] > v) {
      std::swap(arr[--gt], arr[i]);
    } else { // arr[i] == v
      i++;
    }
  }
  std::swap(arr[l], arr[lt]);

  __quickSort3way(arr, l, lt - 1);
  __quickSort3way(arr, gt, r);
}

template <typename Container> void quick_sort_3way(Container &arr) {
  __quickSort3way(arr, 0, arr.size());
}

} // namespace sort
} // namespace algo_and_ds

#endif