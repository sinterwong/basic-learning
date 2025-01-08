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
#include "sort_helper.hpp"
#include <algorithm>
#include <iterator>
#include <memory>
#include <tuple>

namespace algo_and_ds::sort {

template <typename Iter> Iter __partition(Iter first, Iter last) {

  // 选取第一个元素作为基点索引
  Iter p = first;

  // 随机化基点
  std::swap(*first, *(rand() % (last - first + 1) + first));

  // arr[l+1...j] < v; arr[j+1...i) > v
  Iter j = first;

  for (Iter i = first + 1; i != last + 1; i++) {
    if (*i < *p) {
      std::swap(*i, *(++j));
    }
  }
  std::swap(*j, *p);
  return j;
}

template <typename Iter> Iter __partition2way(Iter first, Iter last) {
  // 选取第一个元素作为基点索引
  Iter p = first;

  // 随机化基点
  std::swap(*first, *(rand() % (last - first + 1) + first));

  // arr[l+1...i) < v; arr(j...r] > v
  Iter i = first + 1;
  Iter j = last;

  while (true) {
    while (i <= last && *i < *p) {
      i++;
    }
    while (j >= first + 1 && *j > *p) {
      j--;
    }
    if (i > j) {
      break;
    }
    std::swap(*i, *j);
    i++;
    j--;
  }

  std::swap(*j, *p);
  return j;
}

template <typename Iter> void __quick_sort(Iter first, Iter last) {
  // 当元素数量小于15时就不在往下2分了，直接用插入排序，因为此时更容易有序
  if (last - first <= 15) {
    if (first < last) { // 可能会出现first > last的情况
      insert_sort(first, last + 1);
    }
    return;
  }

  Iter p = __partition(first, last);
  __quick_sort(first, p - 1);
  __quick_sort(p + 1, last);
}

template <typename Iter> void __quick_sort2way(Iter first, Iter last) {
  // 当元素数量小于15时就不在往下2分了，直接用插入排序，因为此时更容易有序
  if (last - first <= 15) {
    if (first < last) { // 可能会出现first > last的情况
      insert_sort(first, last + 1);
    }
    return;
  }

  Iter p = __partition2way(first, last);
  __quick_sort2way(first, p - 1);
  __quick_sort2way(p + 1, last);
}

template <typename Iter> void __quick_sort3way(Iter first, Iter last) {
  // 当元素数量小于15时就不在往下2分了，直接用插入排序，因为此时更容易有序
  if (last - first <= 15) {
    if (first < last) { // 可能会出现first > last的情况
      insert_sort(first, last + 1);
    }
    return;
  }

  // 选取第一个元素作为基点索引
  Iter p = first;

  // 随机化基点
  std::swap(*first, *(rand() % (last - first + 1) + first));

  Iter lt = first;    // arr[l+1...lt] < v
  Iter gt = last + 1; // arr[gt...r] > v
  Iter i = first + 1; // arr[lt+1...i) == v

  while (i < gt) { // 当i和gt相遇时，说明已经遍历完了
    if (*i < *p) {
      std::swap(*i, *(++lt));
      i++;
    } else if (*i > *p) {
      std::swap(*i, *(--gt));
    } else {
      i++;
    }
  }

  std::swap(*p, *lt);

  __quick_sort3way(first, lt - 1);
  __quick_sort3way(gt, last);
}

template <typename Iter> void quick_sort(Iter first, Iter last) {
  srand(time(nullptr));
  __quick_sort(first, last - 1);
}

template <typename Iter> void quick_sort2way(Iter first, Iter last) {
  srand(time(nullptr));
  __quick_sort2way(first, last - 1);
}

template <typename Iter> void quick_sort3way(Iter first, Iter last) {
  srand(time(nullptr));
  __quick_sort3way(first, last - 1);
}

} // namespace algo_and_ds::sort

#endif