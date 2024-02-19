/**
 * @file binary_search.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __AADS_BINARY_SEARCH_HPP_
#define __AADS_BINARY_SEARCH_HPP_
namespace algo_and_ds::algo {
template <typename T, typename Iter>
inline Iter binary_search(Iter begin, Iter end, T target) {
  // Handle empty range explicitly
  if (begin == end)
    return end;

  // 在[l....r)的范围里寻着target，这是循环不变量，后续的数据更新应该遵循
  Iter l = begin;
  Iter r = end; // Use one-past-the-end as the right boundary

  while (l < r) {
    Iter mid = l + (r - l) / 2; // Safe for random access iterators

    if (*mid == target) {
      return mid;
    }

    if (*mid < target) {
      l = mid + 1;
    } else {
      r = mid; // 依然是[l...r)
    }
  }

  return end; // Target not found
}
} // namespace algo_and_ds::algo

#endif