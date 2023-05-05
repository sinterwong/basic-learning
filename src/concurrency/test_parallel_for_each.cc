#include "parallel_for_each.hpp"

int main() {
  std::vector<int> data = {1, 2, 3, 4};
  concurrency::parallel_for_each_async(data.begin(), data.end(),
                                       [](int &it) { it += 1; });

  concurrency::parallel_for_each(data.begin(), data.end(),
                                 [](int &it) { it += 1; });

  for (auto &d : data) {
    std::cout << d << ", ";
  }
  std::cout << std::endl;
  return 0;
}