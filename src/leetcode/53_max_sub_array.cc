/**
 * @file 53_max_sub_array.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an integer array nums, find the subarray with the largest sum,
 * and return its sum.
 * @version 0.1
 * @date 2024-10-14
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <gtest/gtest.h>
#include <limits>

using namespace std;

class Solution {
public:
  // int maxSubArray(vector<int> &nums) {
  //   if (nums.empty()) {
  //     return 0;
  //   }

  //   // 暴力枚举解法（列举出所有subArray的情况，取最大）
  //   int max_so_far = std::numeric_limits<int>::min();

  //   // 遍历所有可能的subArray起始位置
  //   for (int i = 0; i < nums.size(); ++i) {
  //     // 遍历所有可能的subArray结束位置
  //     for (int j = i; j < nums.size(); ++j) {
  //       // 计算每个subArray的和
  //       int current_max = 0;
  //       for (int k = i; k <= j; ++k) {
  //         current_max += nums[k];
  //       }
  //       max_so_far = max(max_so_far, current_max);
  //     }
  //   }
  //   return max_so_far;
  // }

  // int maxSubArray(vector<int> &nums) {
  //   // 使用动态规划求解
  //   /**
  //    * 动态规划的核心在于避免重复计算。
  //    通过将原问题分解成子问题，并将子问题的解存储下来，再利用子问题的重叠性递推后面的结果，可以避免重复计算相同的子问题。

  //    * 子问题：以nums[i]结尾的最大子数组和

  //    * 状态定义：dp[i]表示以nums[i]结尾的最大子数组和

  //    * 状态转移方程：dp[i]=max(nums[i],dp[i-1]+nums[i])，这是动态规划的核心！
  //    它表示：以nums[i]结尾的最大子数组和，要么是nums[i]本身(如果之前的子数组和是负数，就丢弃之前的，从当前元素重新开始)，
  //    要么是前面以nums[i-1] 结尾的最大子数组和加上 nums[i]。

  //    * 初始状态：dp[0]=nums[0]，因为以nums[0]结尾的最大子数组和就是nums[0]本身。

  //    * 自底向上求解：从i=1开始，逐步计算dp[i]，直到dp[n-1]。在这个过程中，global_max始终记录着目前为止的最大子数组和
  //    */
  //   int n = nums.size();
  //   if (n == 0)
  //     return 0;

  //   int global_max = numeric_limits<int>::min();

  //   // dp表示所有的子问题的解，即所有以nums[i]结尾的最大子数组和
  //   vector<int> dp(n);

  //   for (int i = 1; i < n; ++i) {
  //     // dp[i] 表示以 nums[i] 结尾的最大子数组和
  //     dp[i] = max(nums[i], dp[i - 1] + nums[i]);
  //     global_max = max(global_max, dp[i]);
  //   }

  //   return global_max;
  // }

  int maxSubArray(vector<int> &nums) {
    // 动态规划优化版（优化了空间）
    if (nums.empty()) {
      return 0;
    }

    int current_max = 0;

    int global_max = numeric_limits<int>::min();

    for (int num : nums) {
      current_max += num;
      global_max = max(global_max, current_max);

      if (current_max < 0) {
        current_max = 0;
      }
    }
    return global_max;
  }
};

TEST(MaxSubArrayTest, Normal) {
  vector<int> nums = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
  int k = Solution().maxSubArray(nums);
  ASSERT_EQ(k, 6);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
