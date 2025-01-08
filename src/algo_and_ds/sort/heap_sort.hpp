/**
 * @file heap_sort.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * 堆排序，O(nlogn)
 * @version 0.1
 * @date 2022-09-10
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __AADS_HEAP_SORT_HPP_
#define __AADS_HEAP_SORT_HPP_
#include "min_heap.hpp"
#include <iterator>
#include <memory>

namespace algo_and_ds::sort {

template <typename Iter> void heap_sort1(Iter first, Iter last) {
  using ValType = typename std::iterator_traits<Iter>::value_type;
  tree::MinHeap<ValType> minHeap;
  for (auto iter = first; iter != last; ++iter) {
    minHeap.insert(*iter);
  }

  for (auto iter = first; iter != last; ++iter) {
    *iter = minHeap.extractMin();
  }
}

} // namespace algo_and_ds::sort

#endif