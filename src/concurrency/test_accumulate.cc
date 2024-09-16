#include "parallel_accumulate_exception_safe.hpp"
#include <iostream>

int main() {
  std::vector<int> data = {1, 2, 3, 4};
  int sum = my_concurrency::parallel_accumulate_exception_safe2(data.begin(),
                                                             data.end(), 0);

  std::vector<int> data2 = {8, 8, 8, 8};
  sum = my_concurrency::parallel_accumulate_exception_safe(data2.begin(),
                                                        data2.end(), sum);
  std::cout << "sum: " << sum << std::endl;
  return 0;
}