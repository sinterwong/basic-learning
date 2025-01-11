/**
 * @file 209_min_size_subarray_sum.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an array of positive integers nums and a positive integer
target, return the minimal length of a subarray whose sum is greater than or
equal to target. If there is no such subarray, return 0 instead.
 * @version 0.1
 * @date 2024-10-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  // int minSubArrayLen(int target, vector<int> &nums) {
  //   // 暴力解法(O(N^2))
  //   int minLen = nums.size() + 1;
  //   for (int i = 0; i < nums.size(); ++i) {
  //     int sum = 0;
  //     for (int j = i; j < nums.size(); ++j) {
  //       sum += nums[j];
  //       if (sum >= target) {
  //         minLen = min(minLen, j - i + 1);
  //       }
  //     }
  //   }
  //   if (minLen > nums.size()) {
  //     return 0;
  //   }
  //   return minLen;
  // }

  int minSubArrayLen(int target, vector<int> &nums) {
    // 滑动窗口解法（O(N))

    // 滑动窗口范围：[l....r]
    int l = 0;
    int r = -1;
    int sum = 0;
    int minLen = nums.size() + 1;

    while (l < nums.size()) {
      if (r + 1 < nums.size() && sum < target) {
        sum += nums[++r];
      } else {
        sum -= nums[l++];
      }

      if (sum >= target) {
        minLen = min(minLen, r - l + 1);
      }
    }
    if (minLen > nums.size()) {
      return 0;
    }
    return minLen;
  }
};

TEST(MinSizeSubarraySumTest, Normal) {
  vector<int> nums = {2, 3, 1, 2, 4, 3};
  int target = 7;
  ASSERT_EQ(Solution().minSubArrayLen(target, nums), 2);

  nums = {1, 4, 4};
  target = 4;
  ASSERT_EQ(Solution().minSubArrayLen(target, nums), 1);

  nums = {1, 1, 1, 1, 1, 1, 1, 1};
  target = 11;
  ASSERT_EQ(Solution().minSubArrayLen(target, nums), 0);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
