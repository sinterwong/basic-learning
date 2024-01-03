/**
 * @file merge_sort.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * 归并排序，O(nlogn)排序算法，向下2分，到底之后再向上合并，向上合并的过程中看谁大就先放谁。
 向下2分时时间复杂度为O(logn)，每次都需要考虑到所有的元素，复杂度为O(n)，因此为O(nlogn)。
 * @version 0.1
 * @date 2022-09-10
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __AADS_MERGE_SORT_HPP_
#define __AADS_MERGE_SORT_HPP_
#include "insert_sort.hpp"
#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>

namespace algo_and_ds::sort {

template <typename Container>
void __merge(Container &arr, int l, int mid, int r) {
  using valType =
      typename std::iterator_traits<typename Container::iterator>::value_type;
  valType aux[r - l + 1];

  for (int i = l; i <= r; i++) {
    aux[i - l] = arr[i]; // 复制一份数组，这样就可以放心的修改原始数据了
  }
  int i = l;       // 初始化左树
  int j = mid + 1; // 初始化右树
  for (int k = l; k <= r; k++) {
    if (i > mid) {
      // 意味着左边已经完事了，可以直接把右边的放进去就行
      arr[k] = aux[j - l];
      j++;
    } else if (j > r) {
      // 意味着右边已经完事了，直接把左边的放进去就行
      arr[k] = aux[i - l];
      i++;
    } else if (aux[i - l] < aux[j - l]) {
      // 写入左边的
      arr[k] = aux[i - l];
      i++;
    } else {
      // 写入右边的
      arr[k] = aux[j - l];
      j++;
    }
  }
}

template <typename Container> void __mergeSort(Container &arr, int l, int r) {

  // if (l <= r) {
  if (r - l <= 15) {
    // 当元素数量小于15时就不在往下2分了，直接用插入排序，因为此时更容易有序
    insert_sort(arr, l, r);
    return;
  }

  int mid = (l + r) / 2;
  __mergeSort(arr, l, mid);
  __mergeSort(arr, mid + 1, r);
  // 当左边的最大比右边的最小还要小或相等的情况下就不需要归并了
  if (arr[mid] > arr[mid + 1]) {
    __merge(arr, l, mid, r);
  }
}

template <typename Container> void merge_sort(Container &arr) {
  __mergeSort(arr, 0, arr.size() - 1);
}

template <typename Container> void merge_sort_bu(Container &arr) {
  int n = arr.size();
  for (int sz = 1; sz <= n; sz += sz) {
    for (int i = 0; i + sz < n; i += sz + sz) {
      if (arr[i + sz - 1] > arr[i + sz]) {
        __merge(arr, i, i + sz - 1, std::min(i + sz + sz - 1, n - 1));
      }
    }
  }
}

template <typename Iter> void __merge(Iter first, Iter mid, Iter last) {

  // 辅助数组，用于存放合并后的数组，因此空间复杂度为O(n)
  std::vector<typename Iter::value_type> aux(first, last + 1);

  int mid_index = mid - first;

  auto i = aux.begin();
  auto j = aux.begin() + mid_index + 1;

  // k为当前元素的索引
  for (auto k = first; k <= last; ++k) {
    if (i > aux.begin() + mid_index) {
      *k = *j;
      ++j;
    } else if (j == aux.end()) {
      *k = *i;
      ++i;
    } else if (*i < *j) {
      *k = *i;
      ++i;
    } else {
      *k = *j;
      ++j;
    }
  }
}

template <typename Iter> void __merge_sort(Iter first, Iter last) {
  // if (first >= last) {
  //   return;
  // }

  // 当元素数量小于15时就不在往下2分了，直接用插入排序，因为此时更容易有序，且数量很小即使无序插入排序也不会很慢
  if (last - first <= 15) {
    // 插入排序的定义中，last是最后一个元素的下一个位置，需要+1，否则循环条件会出错
    insert_sort(first, last + 1);
    return;
  }

  auto mid = first + (last - first) / 2;
  __merge_sort(first, mid);
  __merge_sort(mid + 1, last);

  // 当左边的最大比右边的最小还要小或相等的情况下就不需要归并了，因为此时已经有序了
  if (*mid > *(mid + 1))
    __merge(first, mid, last);
}

template <typename Iter> void merge_sort(Iter first, Iter last) {
  // 递归使用归并排序，对arr[l...r]的范围进行排序
  __merge_sort(first, last - 1); // last是最后一个元素的下一个位置，需要-1
}

template <typename Iter> void merge_sort_bu(Iter first, Iter last) {
  int n = last - first;
  // sz为子数组的大小，第一次是2^0，第二次是2^1，第三次是2^2，直到2^k > n
  for (int sz = 1; sz <= n; sz += sz) {
    // 子数组的索引范围为[i, i+sz-1]和[i+sz, i+sz+sz-1]
    for (int i = 0; i + sz < n; i += sz + sz) {
      if (*(first + i + sz - 1) > *(first + i + sz)) {
        __merge(first + i, first + i + sz - 1,
                first + std::min(i + sz + sz - 1, n - 1));
      }
    }
  }
}

} // namespace algo_and_ds::sort

#endif