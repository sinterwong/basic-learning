/**
 * @file 217_contains_duplicate.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an integer array nums, return true if any value appears at least
 twice in the array, and return false if every element is distinct.

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
  bool containsDuplicate(vector<int> &nums) {
    unordered_set<int> numsMap;
    for (int i = 0; i < nums.size(); ++i) {
      if (numsMap.find(nums[i]) != numsMap.end()) {
        return true;
      }
      numsMap.insert(nums[i]);
    }
    return false;
  }
};

TEST(ContainsDuplicateTest, Normal) {
  vector<int> nums = {1, 2, 3, 1};
  ASSERT_TRUE(Solution().containsDuplicate(nums));

  nums = {1, 2, 3, 4};
  ASSERT_FALSE(Solution().containsDuplicate(nums));

  nums = {1, 1, 1, 3, 3, 4, 3, 2, 4, 2};
  ASSERT_TRUE(Solution().containsDuplicate(nums));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
