/**
 * @file 88_merge_sorted_array.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief You are given two integer arrays nums1 and nums2, sorted in
non-decreasing order, and two integers m and n, representing the number of
elements in nums1 and nums2 respectively.
Merge nums1 and nums2 into a single array sorted in non-decreasing order.
The final sorted array should not be returned by the function, but instead be
stored inside the array nums1. To accommodate this, nums1 has a length of m + n,
where the first m elements denote the elements that should be merged, and the
last n elements are set to 0 and should be ignored. nums2 has a length of n.
 * @version 0.1
 * @date 2024-10-16
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  void merge(vector<int> &nums1, int m, vector<int> &nums2, int n) {
    vector<int> aux(nums1.size());
    int i = 0; // 维护nums1的索引
    int j = 0; // 维护nums2的索引
    for (int k = 0; k < aux.size(); ++k) {
      if (i >= m) {
        aux[k] = nums2[j++];
      } else if (j >= n) {
        aux[k] = nums1[i++];
      } else if (nums1[i] < nums2[j]) {
        aux[k] = nums1[i++];
      } else {
        aux[k] = nums2[j++];
      }
    }
    nums1 = aux;
  }
};

TEST(MergeSortedArrayTest, Normal) {
  vector<int> nums1 = {1, 2, 3, 0, 0, 0};
  int m = 3;
  vector<int> nums2 = {2, 5, 6};

  int n = 3;
  Solution().merge(nums1, m, nums2, n);

  ASSERT_EQ(nums1, vector<int>({1, 2, 2, 3, 5, 6}));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
