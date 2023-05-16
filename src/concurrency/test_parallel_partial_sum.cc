/**
 * @file test_parallel_partial_sum.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-05-16
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "parallel_partial_sum.hpp"
#include "utils/time_utils.hpp"
#include <iostream>

int main() {
  auto time = utils::measureTime([]() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    concurrency::parallel_partial_sum(v.begin(), v.end());
    for (auto i : v) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  });
  std::cout << "const time: " << static_cast<double>(time) / 1000 << "ms"
            << std::endl;
}