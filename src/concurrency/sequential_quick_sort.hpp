#ifndef _BASIC_CONCUNNENCY_SEQUENTIAL_QUICK_SORT_HPP_
#define _BASIC_CONCUNNENCY_SEQUENTIAL_QUICK_SORT_HPP_

#include <algorithm>
#include <future>
#include <list>

namespace concurrency {
template <typename T> std::list<T> sequential_quick_sort(std::list<T> input) {
  if (input.empty()) {
    return input;
  }
  std::list<T> result;
  // 取出第一个元素作为基准元素
  result.splice(result.begin(), input, input.begin());
  T const &pivot = *result.begin();

  // partition操作，将小于pivot的值全部放到input内的左边，结束后返回基准值当前的iterator
  auto divide_point = std::partition(input.begin(), input.end(),
                                     [&](T const &t) { return t < pivot; });
  std::list<T> lower_part;
  // 将小于部分的值从input中分割出来后，递归
  lower_part.splice(lower_part.end(), input, input.begin(), divide_point);
  auto new_lower(sequential_quick_sort(std::move(lower_part)));

  // 剩余的都是大于的基准的值，递归
  auto new_higher(sequential_quick_sort(std::move(input)));

  // 结束后
  result.splice(result.end(), new_higher);
  result.splice(result.begin(), new_lower);
  return result;
}

template <typename T> std::list<T> parallel_quick_sort(std::list<T> input) {
  if (input.empty()) {
    return input;
  }
  std::list<T> result;
  result.splice(result.begin(), input, input.begin());
  T const &pivot = *result.begin();
  auto divide_point = std::partition(input.begin(), input.end(),
                                     [&](T const &t) { return t < pivot; });
  std::list<T> lower_part;
  lower_part.splice(lower_part.end(), input, input.begin(), divide_point);
  std::future<std::list<T>> new_lower(
      std::async(&parallel_quick_sort(std::move(lower_part))));
  std::future<std::list<T>> new_higher(parallel_quick_sort(std::move(input)));
  result.splice(result.end(), new_higher);
  result.splice(result.begin(), new_lower.get());
  return result;
}
} // namespace concurrency

#endif
