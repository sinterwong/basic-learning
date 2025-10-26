/**
 * @file 15_3sum.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an integer array nums, return all the triplets [nums[i],
nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] +
nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.
 * @version 0.1
 * @date 2024-10-22
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <algorithm>
#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

using namespace std;

class Solution {
public:
  // vector<vector<int>> threeSum(vector<int> &nums) {
  //   // 暴力枚举解法 O(N^3)
  //   set<vector<int>> allResults;
  //   for (int i = 0; i < nums.size() - 2; ++i) {
  //     for (int j = i + 1; j < nums.size() - 1; ++j) {
  //       for (int k = j + 1; k < nums.size(); ++k) {
  //         if (nums[i] + nums[j] + nums[k] == 0) {
  //           vector<int> ret = {nums[i], nums[j], nums[k]};
  //           sort(ret.begin(), ret.end());
  //           allResults.insert(ret);
  //         }
  //       }
  //     }
  //   }
  //   return vector<vector<int>>(allResults.begin(), allResults.end());;
  // }

  // vector<vector<int>> threeSum(vector<int> &nums) {
  //   // 排序+对撞指针 O(N^2)
  //   set<vector<int>> allResults;
  //   sort(nums.begin(), nums.end());
  //   for (int i = 0; i < nums.size() - 2; ++i) {
  //     int target = 0 - nums[i];

  //     int l = i + 1;
  //     int r = nums.size() - 1;
  //     while (l < r) {
  //       int lastTwoVal = nums[l] + nums[r];
  //       if (lastTwoVal == target) {
  //         allResults.insert({nums[i], nums[l++], nums[r--]});
  //       } else if (lastTwoVal < target) {
  //         l++;
  //       } else {
  //         r--;
  //       }
  //     }
  //   }
  //   return vector<vector<int>>(allResults.begin(), allResults.end());
  // }

  vector<vector<int>> threeSum(vector<int> &nums) {
    // 哈希表 O(N^2)
    set<vector<int>> allResults;
    for (int i = 0; i < nums.size() - 2; ++i) {
      int target = 0 - nums[i];
      unordered_map<int, int> valToIndexMap;
      for (int j = i + 1; j < nums.size(); ++j) {
        int currNum = nums[j];
        int twoTarget = target - currNum;
        if (valToIndexMap.find(twoTarget) != valToIndexMap.end()) {
          vector<int> ret = {nums[i], nums[j], twoTarget};
          sort(ret.begin(), ret.end());
          allResults.insert(ret);
          valToIndexMap.erase(twoTarget);
        } else {
          valToIndexMap[currNum] = j;
        }
      }
    }
    return vector<vector<int>>(allResults.begin(), allResults.end());
  }
};

TEST(ThreeSumTest, Normal) {
  vector<int> nums = {-1, 0, 1, 2, -1, -4};
  auto ret = Solution().threeSum(nums);
  ASSERT_EQ(ret, vector<vector<int>>({{-1, -1, 2}, {-1, 0, 1}}));

  nums = {0, 1, 1};
  ret = Solution().threeSum(nums);
  ASSERT_EQ(ret, vector<vector<int>>({}));

  nums = {0, 0, 0};
  ret = Solution().threeSum(nums);
  ASSERT_EQ(ret, vector<vector<int>>({{0, 0, 0}}));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
