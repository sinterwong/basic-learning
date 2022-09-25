#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include "algo_and_ds/sort_helper.hpp"
#include "algo_and_ds/select_sort.hpp"
#include "algo_and_ds/insert_sort.hpp"
#include "algo_and_ds/merge_sort.hpp"

using namespace algo_and_ds::sort;

int main(int argc, char **argv) {
  const int numElements = 1000000;
  using citer = std::array<int, numElements>::const_iterator;
  using iter = std::array<int, numElements>::iterator;
  using myArray = std::array<int, numElements>;

  // myArray randomArr1;
  // generateRandomArray<myArray>(randomArr1, 10000);
  // myArray randomArr2 = randomArr1;
  // // myArray randomArr3 = randomArr1;
  // // testSort<myArray>("Select sort", select_sort, randomArr5);
  // testSort<myArray>("Insert sort", insert_sort, randomArr1);
  // testSort<myArray>("Merge sort", merge_sort, randomArr2);

  myArray nearlyOrderArray1;
  generateNearlyOrderedArray<myArray>(nearlyOrderArray1, 0);
  myArray nearlyOrderArray2 = nearlyOrderArray1;
  // myArray nearlyOrderArray3 = nearlyOrderArray1;
  // testSort<myArray>("Select sort", select_sort, nearlyOrderArray5);
  testSort<myArray>("Insert sort", insert_sort, nearlyOrderArray1);
  testSort<myArray>("Merge sort", merge_sort, nearlyOrderArray2);

  return 0;
}