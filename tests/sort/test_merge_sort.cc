/**
 * @file test_merge_sort.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-12-31
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <vector>

#include "insert_sort.hpp"
#include "merge_sort.hpp"
#include "sort_helper.hpp"
#include <gtest/gtest.h>

using namespace algo_and_ds::sort;

class MergeSortTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
  int n = 10000;
};

TEST_F(MergeSortTest, Normal) {
  std::vector<int> arrMergeRand, arrMergeBuRand, arrInsertRand;
  arrMergeRand.resize(n);
  generateRandomArray(arrMergeRand.begin(), arrMergeRand.end(), 0, n);
  arrMergeBuRand = arrMergeRand; // copy
  arrInsertRand = arrMergeRand;  // copy

  testSort("merge sort(rand data): ", arrMergeRand.begin(), arrMergeRand.end(),
           [](auto first, auto last) { merge_sort(first, last); });

  testSort("merge sort bottom up(rand data): ", arrMergeBuRand.begin(),
           arrMergeBuRand.end(),
           [](auto first, auto last) { merge_sort_bu(first, last); });

  testSort("Insertion sort(rand data): ", arrInsertRand.begin(),
           arrInsertRand.end(),
           [](auto first, auto last) { insert_sort(first, last); });
}

TEST_F(MergeSortTest, NearlyOrdered) {
  std::vector<int> arrMergeNear, arrMergeBuNear, arrInsertNear;
  arrMergeNear.resize(n);
  generateNearlyOrderedArray(arrMergeNear.begin(), arrMergeNear.end(), 10);
  arrMergeBuNear = arrMergeNear; // copy
  arrInsertNear = arrMergeNear;  // copy

  testSort("merge sort(nearly ordered data): ", arrMergeNear.begin(),
           arrMergeNear.end(),
           [](auto first, auto last) { merge_sort(first, last); });

  testSort("merge sort bottom up(nearly ordered data): ",
           arrMergeBuNear.begin(), arrMergeBuNear.end(),
           [](auto first, auto last) { merge_sort_bu(first, last); });

  testSort("Insertion sort(nearly ordered data): ", arrInsertNear.begin(),
           arrInsertNear.end(),
           [](auto first, auto last) { insert_sort(first, last); });
}
