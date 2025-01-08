/**
 * @file test_selection_sort.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-12-30
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "insert_sort.hpp"
#include "select_sort.hpp"
#include "sort_helper.hpp"
#include <gtest/gtest.h>

using namespace algo_and_ds::sort;

class ON2SortTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
  int n = 10000;
};

TEST_F(ON2SortTest, Normal) {
  std::vector<int> arrInsertionRand, arrSelectionRand;
  arrInsertionRand.resize(n);
  generateRandomArray(arrInsertionRand.begin(), arrInsertionRand.end(), 0, n);
  arrSelectionRand = arrInsertionRand;

  // insert sort
  testSort("Insertion sort(rand data): ", arrInsertionRand.begin(),
           arrInsertionRand.end(),
           [](auto first, auto last) { insert_sort(first, last); });

  // select sort
  testSort("Selection sort(rand data): ", arrSelectionRand.begin(),
           arrSelectionRand.end(),
           [](auto first, auto last) { select_sort(first, last); });
}

TEST_F(ON2SortTest, NearlyOrdered) {
  std::vector<int> arrInsertionNear, arrSelectionNear;
  arrInsertionNear.resize(n);
  generateNearlyOrderedArray(arrInsertionNear.begin(), arrInsertionNear.end(),
                             10);
  arrSelectionNear = arrInsertionNear;

  // insert sort
  testSort("Insertion sort(nearly ordered data): ", arrInsertionNear.begin(),
           arrInsertionNear.end(),
           [](auto first, auto last) { insert_sort(first, last); });

  // select sort
  testSort("Selection sort(nearly ordered data): ", arrSelectionNear.begin(),
           arrSelectionNear.end(),
           [](auto first, auto last) { select_sort(first, last); });
}
