/**
 * @file 18_4sum.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an array nums of n integers, return an array of all the unique
quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:

0 <= a, b, c, d < n
a, b, c, and d are distinct.
nums[a] + nums[b] + nums[c] + nums[d] == target
You may return the answer in any order.
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
  vector<vector<int>> fourSum(vector<int> &nums, int target) {
    int n = nums.size();
    vector<vector<int>> result;
    if (n < 4)
      return result;

    sort(nums.begin(), nums.end());

    // 计算并存储后两位的和
    unordered_map<int64_t, vector<pair<int, int>>> twoSumMap;
    for (int i = 2; i < n - 1; ++i) {
      for (int j = i + 1; j < n; ++j) {
        twoSumMap[(int64_t)nums[i] + nums[j]].push_back({i, j});
      }
    }

    // 计算前两位的和
    for (int i = 0; i < n - 3; ++i) {
      if (i > 0 && nums[i] == nums[i - 1])
        continue;
      for (int j = i + 1; j < n - 2; ++j) {
        if (j > i + 1 && nums[j] == nums[j - 1])
          continue;
        int64_t complement = (int64_t)target - nums[i] - nums[j];
        if (twoSumMap.count(complement)) {
          for (auto &p : twoSumMap[complement]) {
            int k = p.first;
            int l = p.second;
            if (k > j) {
              vector<int> quadruplet = {nums[i], nums[j], nums[k], nums[l]};
              result.push_back(quadruplet);
            }
          }
        }
      }
    }
    result.erase(unique(result.begin(), result.end()), result.end());
    return result;
  }
};

TEST(FourSumTest, Normal) {
  vector<int> nums = {1, 0, -1, 0, -2, 2};
  int target = 0;
  auto ret = Solution().fourSum(nums, target);
  ASSERT_EQ(
      ret, vector<vector<int>>({{-2, -1, 1, 2}, {-2, 0, 0, 2}, {-1, 0, 0, 1}}));

  nums = {2, 2, 2, 2, 2};
  target = 8;
  ret = Solution().fourSum(nums, target);
  ASSERT_EQ(ret, vector<vector<int>>({{2, 2, 2, 2}}));

  nums = {-3, -2, -1, 0, 0, 1, 2, 3};
  target = 0;
  ret = Solution().fourSum(nums, target);
  // [[-3,-2,2,3],[-3,-1,1,3],[-3,0,0,3],[-3,0,1,2],[-2,-1,0,3],[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
  ASSERT_EQ(ret, vector<vector<int>>({{-3, -2, 2, 3},
                                      {-3, -1, 1, 3},
                                      {-3, 0, 0, 3},
                                      {-3, 0, 1, 2},
                                      {-2, -1, 0, 3},
                                      {-2, -1, 1, 2},
                                      {-2, 0, 0, 2},
                                      {-1, 0, 0, 1}}));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
