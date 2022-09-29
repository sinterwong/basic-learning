#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include "algo_and_ds/sort_helper.hpp"
#include "algo_and_ds/select_sort.hpp"
#include "algo_and_ds/insert_sort.hpp"
#include "algo_and_ds/merge_sort.hpp"
#include "algo_and_ds/quick_sort.hpp"

using namespace algo_and_ds::sort;

int main(int argc, char *argv[]) {

  const int numElements = 209399;
  using citer = std::array<int, numElements>::const_iterator;
  using iter = std::array<int, numElements>::iterator;
  using myArray = std::array<int, numElements>;

  myArray randomArr1;
  generateRandomArray<myArray>(randomArr1, 10000);
  myArray randomArr2 = randomArr1;
  myArray randomArr3 = randomArr1;
  // testSort<myArray>("Select sort", select_sort, randomArr15);
  // testSort<myArray>("Insert sort", insert_sort, randomArr14);
  // testSort<myArray>("Merge sort", merge_sort, randomArr2);
  testSort<myArray>("Merge bu sort", merge_sort_bu, randomArr1);
  testSort<myArray>("Quick sort", quick_sort, randomArr2);
  testSort<myArray>("Quick sort 2 way", quick_sort_2way, randomArr3);

  myArray nearlyOrderArray1;
  generateNearlyOrderedArray<myArray>(nearlyOrderArray1, 10);
  myArray nearlyOrderArray2 = nearlyOrderArray1;
  myArray nearlyOrderArray3 = nearlyOrderArray1;
  testSort<myArray>("Merge bu sort", merge_sort_bu, nearlyOrderArray1);
  testSort<myArray>("Quick sort", quick_sort, nearlyOrderArray2);
  testSort<myArray>("Quick sort 2 way", quick_sort_2way, nearlyOrderArray3);

  myArray repeatArr1;
  generateRandomArray<myArray>(repeatArr1, 10);
  myArray repeatArr2 = repeatArr1;
  myArray repeatArr3 = repeatArr1;
  testSort<myArray>("Merge bu sort", merge_sort_bu, repeatArr1);
  // testSort<myArray>("Quick sort", quick_sort, repeatArr2);
  testSort<myArray>("Quick sort 2 way", quick_sort_2way, repeatArr3);

  return 0;
}