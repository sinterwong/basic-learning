/**
 * @file test_parallel_for_find.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-05-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "parallel_for_find.hpp"
#include "utils/time_utils.hpp"
#include <algorithm>

int main(int argc, char const *argv[]) {
  /* 测试10万个随机元素中查找某值 */
  std::vector<int> v(100000);
  std::generate(v.begin(), v.end(), []() { return rand() % 1000; });
  std::for_each_n(v.begin(), 10, [](auto &p) { std::cout << p << std::endl; });
  auto time = utils::measureTime([&]() {
    auto it = concurrency::parallel_for_find(v.begin(), v.end(), 50);
    if (it == v.end()) {
      std::cout << "not found" << std::endl;
      return;
    }
    std::cout << "find: " << *it << std::endl;
  });
  std::cout << "cost time: " << static_cast<double>(time) / 1000 << "ms"
            << std::endl;

  auto time2 = utils::measureTime([&]() {
    auto it = concurrency::parallel_for_find_async(v.begin(), v.end(), 50);
    if (it == v.end()) {
      std::cout << "not found" << std::endl;
      return;
    }
    std::cout << "find: " << *it << std::endl;
  });
  std::cout << "cost time: " << static_cast<double>(time2) / 1000 << "ms"
            << std::endl;

  return 0;
}