/**
 * @file 454_4sum_II.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given four integer arrays nums1, nums2, nums3, and nums4 all of length
n, return the number of tuples (i, j, k, l) such that:

0 <= i, j, k, l < n
nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0
 * @version 0.1
 * @date 2024-10-23
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <cstdint>
#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  int fourSumCount(vector<int> &nums1, vector<int> &nums2, vector<int> &nums3,
                   vector<int> &nums4) {
    unordered_map<int64_t, int> firstTwoSum;
    for (int i = 0; i < nums1.size(); ++i) {
      for (int j = 0; j < nums2.size(); ++j) {
        firstTwoSum[nums1[i] + nums2[j]]++;
      }
    }

    int ret = 0;

    for (int i = 0; i < nums3.size(); ++i) {
      for (int j = 0; j < nums4.size(); ++j) {
        int64_t target = 0 - nums3[i] - nums4[j];
        if (firstTwoSum.find(target) != firstTwoSum.end()) {
          ret += firstTwoSum[target];
        }
      }
    }
    return ret;
  }
};

TEST(FourSumIITest, Normal) {
  vector<int> nums1 = {1, 2};
  vector<int> nums2 = {-2, -1};
  vector<int> nums3 = {-1, 2};
  vector<int> nums4 = {0, 2};
  ASSERT_EQ(Solution().fourSumCount(nums1, nums2, nums3, nums4), 2);

  nums1 = {0};
  nums2 = {0};
  nums3 = {0};
  nums4 = {0};
  ASSERT_EQ(Solution().fourSumCount(nums1, nums2, nums3, nums4), 1);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
