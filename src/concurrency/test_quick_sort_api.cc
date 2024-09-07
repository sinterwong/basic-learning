#include "thread_pool_quick_sort.hpp"
#include <iostream>

int main() {
  std::list<int> data = {3, 5, 1, 2, 3432, 564, 1, 23, 4, 34, 234, 2314};

  auto ret = my_concurrency::parallel_quick_sort(data);

  for (auto &r : ret) {
    std::cout << r << ", ";
  }
  std::cout << std::endl;
  return 0;
}