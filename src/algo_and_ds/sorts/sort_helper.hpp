

#include <array>
#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>

namespace algo_and_ds {
namespace sort {

template <typename Iter> void printArray(Iter begin, Iter end) {
  for (auto iter = begin; iter != end; iter++) {
    std::cout << *iter << " ";
  }
  std::cout << std::endl;
}

template <typename ForwardIterator>
bool isSorted(ForwardIterator first, ForwardIterator last) {
  for (auto iter = ++first; iter != last; iter++) {
    if (*iter < *(iter - 1)) {
      return false;
    }
  }
  return true;
}

template <typename Iter>
void generateRandomArray(Iter begin, Iter end, int leftRange, int rightRange) {
  srand(time(nullptr));
  for (auto iter = begin; iter != end; iter++) {
    *iter = static_cast<typename std::iterator_traits<Iter>::value_type>(
        rand() % (rightRange - leftRange) + leftRange);
  }
}

template <typename Iter> void generateRange(Iter begin, Iter end) {
  for (auto iter = begin; iter != end; iter++) {
    *iter = static_cast<typename std::iterator_traits<Iter>::value_type>(iter -
                                                                         begin);
  }
}

template <typename Iter>
void generateNearlyOrderedArray(Iter begin, Iter end, int swapTimes) {
  // 生成完全有序数组
  generateRange(begin, end);
  srand(time(nullptr));
  for (int i = 0; i < swapTimes; ++i) {
    int x = rand() % (end - begin);
    int y = rand() % (end - begin);
    std::swap(*(begin + x), *(begin + y));
  }
}

template <typename Iter, typename Func>
void testSort(std::string name, Iter first, Iter last, Func &&func) {
  auto start = std::chrono::steady_clock::now();
  func(first, last);
  auto end = std::chrono::steady_clock::now();
  assert(isSorted(first, last));
  auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();

  std::cout << name << static_cast<double>(time) / 1000000 << "s" << std::endl;
}

} // namespace sort
} // namespace algo_and_ds