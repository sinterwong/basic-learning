#ifndef _BASIC_CONCUNNENCY_SEQUENTIAL_QUICK_SORT_HPP_
#define _BASIC_CONCUNNENCY_SEQUENTIAL_QUICK_SORT_HPP_

#include <algorithm>
#include <future>
#include <list>

namespace my_concurrency {
template <typename T> std::list<T> sequential_quick_sort(std::list<T> input) {
  if (input.empty()) {
    return input;
  }
  std::list<T> result;
  // splice op: 从 input 中移出input.begin()，插入到result.begin()位置
  result.splice(result.begin(), input, input.begin());
  T const &pivot = *result.begin();

  // partition op (O(n)):
  // 将input.begin()至input.end()直接的元素根据predicate划分成两部分，返回值指向第一个不符合条件的元素
  auto divide_point = std::partition(
      input.begin(), input.end(), [&pivot](T const &v) { return v < pivot; });

  std::list<T> lower_part;
  // splice op: 从 input 中移出[input.begin(), divide_point)到lower_part的尾部
  lower_part.splice(lower_part.end(), input, input.begin(), divide_point);
  auto new_lower(sequential_quick_sort(std::move(lower_part)));

  auto new_higher(sequential_quick_sort(std::move(input)));

  result.splice(result.end(), new_higher);
  result.splice(result.begin(), new_lower);
  return result;
}

template <typename T>
std::list<T> async_parallel_quick_sort(std::list<T> input) {
  if (input.empty()) {
    return input;
  }
  std::list<T> result;
  result.splice(result.begin(), input, input.begin());
  T const &pivot = *result.begin();
  auto divide_point = std::partition(
      input.begin(), input.end(), [&pivot](T const &v) { return v < pivot; });
  std::list<T> lower_part;
  lower_part.splice(lower_part.end(), input, input.begin(), divide_point);
  std::future<std::list<T>> new_lower(
      std::async(&async_parallel_quick_sort<T>, std::move(lower_part)));
  std::list<T> new_higher(async_parallel_quick_sort(std::move(input)));
  result.splice(result.end(), new_higher);
  result.splice(result.begin(), new_lower.get());
  return result;
}
} // namespace my_concurrency

#endif
