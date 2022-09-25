#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include "algo_and_ds/sort_helper.hpp"
#include "algo_and_ds/select_sort.hpp"
using namespace algo_and_ds::sort;

int main(int argc, char **argv) {
  const int numElements = 10000;
  using citer = std::array<int, numElements>::const_iterator;
  using iter = std::array<int, numElements>::iterator;
  using myArray = std::array<int, numElements>;

  myArray arr;
  generateRandomArray<myArray>(arr, 10000);
  testSort<myArray>("Select sort", select_sort, arr);

  myArray arr2;
  generateNearlyOrderedArray<myArray>(arr2, 1);
  return 0;
}