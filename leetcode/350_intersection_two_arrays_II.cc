/**
 * @file 350_intersection_two_arrays_II.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given two integer arrays nums1 and nums2, return an array of their
 * intersection. Each element in the result must appear as many times as it
 * shows in both arrays and you may return the result in any order.
 * @version 0.1
 * @date 2024-10-21
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  vector<int> intersect(vector<int> &nums1, vector<int> &nums2) {

    vector<int> rets;
    map<int, int> nums1Map;
    for (auto n : nums1) {
      nums1Map[n]++;
    }

    for (auto n : nums2) {
      if (nums1Map.find(n) != nums1Map.end() && nums1Map[n] > 0) {
        rets.push_back(n);
        nums1Map[n]--;
      }
    }
    return rets;
  }
};

TEST(IntersectionTwoArraysIITest, Normal) {
  vector<int> nums1 = {1, 2, 2, 1};
  vector<int> nums2 = {2, 2};
  auto ret = Solution().intersect(nums1, nums2);
  ASSERT_EQ(ret, vector<int>({2, 2}));

  nums1 = {4, 9, 5};
  nums2 = {9, 4, 9, 8, 4};
  ret = Solution().intersect(nums1, nums2);
  ASSERT_EQ(ret, vector<int>({9, 4}));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
