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

  Iter l = begin;
  Iter r = end; // Use one-past-the-end as the right boundary

  while (l < r) {
    Iter mid = l + (r - l) / 2; // Safe for random access iterators

    if (*mid == target) {
      return mid;
    } else if (*mid < target) {
      l = mid + 1;
    } else {
      r = mid;
    }
  }

  return end; // Target not found
}
} // namespace algo_and_ds::algo

#endif