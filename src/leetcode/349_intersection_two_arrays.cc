/**
 * @file 349_intersection_two_arrays.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given two integer arrays nums1 and nums2, return an array of their
intersection. Each element in the result must be unique and you may return the
result in any order.
 * @version 0.1
 * @date 2024-10-21
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <gtest/gtest.h>
#include <vector>

using namespace std;

class Solution {
public:
  vector<int> intersection(vector<int> &nums1, vector<int> &nums2) {
    set<int> retsSet;

    set<int> nums1Set(nums1.begin(), nums1.end());

    for (int i = 0; i < nums2.size(); ++i) {
      if (nums1Set.find(nums2[i]) != nums1Set.end()) {
        retsSet.insert(nums2[i]);
      }
    }
    return vector<int>(retsSet.begin(), retsSet.end());
  }
};

TEST(IntersectionTwoArraysTest, Normal) {
  vector<int> nums1 = {1, 2, 2, 1};
  vector<int> nums2 = {2, 2};
  auto ret = Solution().intersection(nums1, nums2);
  ASSERT_EQ(ret, vector<int>({2}));

  nums1 = {4, 9, 5};
  nums2 = {9, 4, 9, 8, 4};
  ret = Solution().intersection(nums1, nums2);
  ASSERT_EQ(ret, vector<int>({4, 9}));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
