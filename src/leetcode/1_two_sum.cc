/**
 * @file 1_two_sum.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an array of integers nums and an integer target, return indices
of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not
use the same element twice.

You can return the answer in any order.
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
  vector<int> twoSum(vector<int> &nums, int target) {
    unordered_map<int, int> numsIndexMap;
    for (int i = 0; i < nums.size(); ++i) {
      int currVal = nums[i];
      if (numsIndexMap.find(target - currVal) != numsIndexMap.end()) {
        return {numsIndexMap[target - currVal], i};
      }
      numsIndexMap[nums[i]] = i;
    }
    return {};
  }
};

TEST(TwoSumTest, Normal) {
  vector<int> nums = {2, 7, 11, 15};
  int target = 9;
  auto ret = Solution().twoSum(nums, target);
  ASSERT_EQ(ret, vector<int>({0, 1}));

  nums = {3, 2, 4};
  target = 6;
  ret = Solution().twoSum(nums, target);
  ASSERT_EQ(ret, vector<int>({1, 2}));

  nums = {3, 3};
  target = 6;
  ret = Solution().twoSum(nums, target);
  ASSERT_EQ(ret, vector<int>({0, 1}));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
