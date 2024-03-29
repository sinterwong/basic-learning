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

namespace algo_and_ds::sort {

template <typename Iter> void insert_sort(Iter first, Iter last) {
  using valType = typename std::iterator_traits<Iter>::value_type;
  // [first, i) 有序，因此从i开始遍历
  for (auto i = first + 1; i != last; ++i) {
    valType temp = *i;
    Iter j;
    for (j = i; j != first && *(j - 1) > temp; --j) {
      *j = *(j - 1);
    }
    *j = temp;
  }
}

// 所有的迭代任务都可以用递归来完成，反之亦然
template <typename Iter> void insert_sort_recursive(Iter first, Iter last) {
  last--; // 令last指向最后一个元素
  using valType = typename std::iterator_traits<Iter>::value_type;
  // first == last 时，只有一个元素，不需要排序
  if (first == last) {
    return;
  }

  insert_sort_recursive(first, last);

  valType temp = *(last);
  Iter j;
  for (j = last; j != first && *(j - 1) > temp; --j) {
    *j = *(j - 1);
  }
  *j = temp;
}
} // namespace algo_and_ds::sort

#endif