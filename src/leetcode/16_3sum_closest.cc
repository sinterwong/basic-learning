/**
 * @file 16_3sum_closest.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an integer array nums of length n and an integer target, find
three integers in nums such that the sum is closest to target.

Return the sum of the three integers.

You may assume that each input would have exactly one solution.
 * @version 0.1
 * @date 2024-10-22
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <gtest/gtest.h>
#include <vector>

using namespace std;

class Solution {
public:
  // int threeSumClosest(vector<int> &nums, int target) {
  //   // 暴力枚举法：O(N^3)
  //   int closestVal = numeric_limits<int>::max();
  //   for (int i = 0; i < nums.size() - 2; ++i) {
  //     for (int j = i + 1; j < nums.size() - 1; ++j) {
  //       for (int k = j + 1; k < nums.size(); ++k) {
  //         int curVal = nums[i] + nums[j] + nums[k];
  //         if (abs(target - curVal) < abs(closestVal)) {
  //           closestVal = curVal;
  //         }
  //       }
  //     }
  //   }
  //   return closestVal;
  // }

  int threeSumClosest(vector<int> &nums, int target) {
    sort(nums.begin(), nums.end());

    int closestVal = numeric_limits<int>::max();
    int minDis = numeric_limits<int>::max();
    for (int i = 0; i < nums.size() - 2; ++i) {
      int l = i + 1;
      int r = nums.size() - 1;
      while (l < r) {
        int curVal = nums[i] + nums[l] + nums[r];
        if (curVal == target) {
          return target;
        } else if (curVal < target) {
          l++;
        } else {
          r--;
        }
        int curDis = abs(target - curVal);
        if (curDis < minDis) {
          closestVal = curVal;
          minDis = curDis;
        }
      }
    }
    return closestVal;
  }
};

TEST(ThreeSumClosestTest, Normal) {
  vector<int> nums = {-1, 2, 1, -4};
  int target = 1;
  ASSERT_EQ(Solution().threeSumClosest(nums, target), 2);

  nums = {0, 0, 0};
  target = 1;
  ASSERT_EQ(Solution().threeSumClosest(nums, target), 0);

  nums = {0, 1, 2};
  target = 3;
  ASSERT_EQ(Solution().threeSumClosest(nums, target), 3);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
