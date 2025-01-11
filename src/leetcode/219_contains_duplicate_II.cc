/**
 * @file 219_contains_duplicate_II.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an integer array nums and an integer k, return true if there are
 * two distinct indices i and j in the array such that nums[i] == nums[j] and
 * abs(i - j) <= k.
 * @version 0.1
 * @date 2024-10-23
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <gtest/gtest.h>
#include <unordered_set>

using namespace std;

class Solution {
public:
  // bool containsNearbyDuplicate(vector<int> &nums, int k) {
  //   // 暴力解法 O(n ^ 2)
  //   for (int i = 0; i < nums.size() - 1; ++i) {
  //     for (int j = i + 1; j < nums.size(); ++j) {
  //       if (nums[i] == nums[j] && j - i <= k) {
  //         return true;
  //       }
  //     }
  //   }
  //   return false;
  // }
  bool containsNearbyDuplicate(vector<int> &nums, int k) {
    // 滑动窗口 + 查找表 O(n)
    unordered_set<int> kWindow;
    for (int i = 0; i < nums.size(); ++i) {
      if (kWindow.find(nums[i]) != kWindow.end()) {
        return true;
      }

      kWindow.insert(nums[i]);

      if (kWindow.size() > k) {
        kWindow.erase(nums[i - k]);
      }
    }
    return false;
  }
};

TEST(ContainsDuplicateIITest, Normal) {
  vector<int> nums = {1, 2, 3, 1};
  int k = 3;
  ASSERT_TRUE(Solution().containsNearbyDuplicate(nums, k));

  nums = {1, 0, 1, 1};
  k = 1;
  ASSERT_TRUE(Solution().containsNearbyDuplicate(nums, k));

  nums = {1, 2, 3, 1, 2, 3};
  k = 2;
  ASSERT_FALSE(Solution().containsNearbyDuplicate(nums, k));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
