#include <iostream>
#include <vector>
#include "algo_and_ds/select_sort.hpp"
using namespace algo_and_ds::sort;

int main(int argc, char **argv) {
  std::vector<int> arr = {100, 24, 324, 11, 7};
  select_sort(arr);

  for (auto d : arr) {
    std::cout << d << " ";
  }
  std::cout << std::endl;

  return 0;
}