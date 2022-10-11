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
#include <iterator>
#include <memory>

#include "max_heap.hpp"

namespace algo_and_ds {
namespace sort {

template <typename Container> void heap_sort(Container &arr) {
  using valType =
      typename std::iterator_traits<typename Container::iterator>::value_type;
  tree::MaxHeap<valType> maxHeap;
  for (auto &i : arr) {
    maxHeap.insert(i);
  }

  for (int i = arr.size() - 1; i >= 0; --i) {
    arr[i] = maxHeap.extractMax();
  }
}

// template <typename Container> void heapify_sort(Container &arr) {
//   tree::MaxHeap<valType> maxHeap{arr};
//   for (int i = arr.size() - 1; i >= 0; --i) {
//     arr[i] = maxHeap.extractMax();
//   }
// }

} // namespace sort
} // namespace algo_and_ds

#endif