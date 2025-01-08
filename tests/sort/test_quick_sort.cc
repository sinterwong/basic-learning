/**
 * @file test_quick_sort.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-01-03
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "merge_sort.hpp"
#include "quick_sort.hpp"
#include "sort_helper.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace algo_and_ds::sort;

class QuickSortTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
  int n = 1000000;
};

TEST_F(QuickSortTest, Normal) {
  std::vector<int> arrQuickRand, arrQuick2Rand, arrMergeRand;
  arrQuickRand.resize(n);
  generateRandomArray(arrQuickRand.begin(), arrQuickRand.end(), 0, n);
  arrQuick2Rand = arrQuickRand; // copy
  arrMergeRand = arrQuickRand;  // copy

  testSort("quick sort 3way(rand data): ", arrQuickRand.begin(),
           arrQuickRand.end(),
           [](auto first, auto last) { quick_sort3way(first, last); });

  testSort("quick sort 2way(rand data): ", arrQuick2Rand.begin(),
           arrQuick2Rand.end(),
           [](auto first, auto last) { quick_sort2way(first, last); });

  testSort("merge sort(rand data): ", arrMergeRand.begin(), arrMergeRand.end(),
           [](auto first, auto last) { merge_sort(first, last); });
}

TEST_F(QuickSortTest, NearlyOrdered) {
  std::vector<int> arrQuickNear, arrQuick2Near, arrMergeNear;
  arrQuickNear.resize(n);
  generateNearlyOrderedArray(arrQuickNear.begin(), arrQuickNear.end(), 100);
  arrQuick2Near = arrQuickNear; // copy
  arrMergeNear = arrQuickNear;  // copy

  testSort("quick sort 3way(near ordered data): ", arrQuickNear.begin(),
           arrQuickNear.end(),
           [](auto first, auto last) { quick_sort3way(first, last); });

  testSort("quick sort 2way(near ordered data): ", arrQuick2Near.begin(),
           arrQuick2Near.end(),
           [](auto first, auto last) { quick_sort2way(first, last); });
  testSort("merge sort(near ordered data): ", arrMergeNear.begin(),
           arrMergeNear.end(),
           [](auto first, auto last) { merge_sort(first, last); });
}

TEST_F(QuickSortTest, Repeat) {
  std::vector<int> arrQuickRand, arrQuick2Rand, arrMergeRand;
  arrQuickRand.resize(n);
  generateRandomArray(arrQuickRand.begin(), arrQuickRand.end(), 0, 10);
  arrQuick2Rand = arrQuickRand; // copy
  arrMergeRand = arrQuickRand;  // copy

  testSort("quick sort 3way(repeat data): ", arrQuickRand.begin(),
           arrQuickRand.end(),
           [](auto first, auto last) { quick_sort3way(first, last); });

  testSort("quick sort 2way(repeat data): ", arrQuick2Rand.begin(),
           arrQuick2Rand.end(),
           [](auto first, auto last) { quick_sort2way(first, last); });

  testSort("merge sort(repeat data): ", arrMergeRand.begin(),
           arrMergeRand.end(),
           [](auto first, auto last) { merge_sort(first, last); });
}
