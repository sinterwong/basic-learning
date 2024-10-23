/**
 * @file 220_contains_duplicate_III.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief You are given an integer array nums and two integers indexDiff and
valueDiff.

Find a pair of indices (i, j) such that:

i != j,
abs(i - j) <= indexDiff.
abs(nums[i] - nums[j]) <= valueDiff, and
Return true if such pair exists or false otherwise.
 * @version 0.1
 * @date 2024-10-23
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  bool containsNearbyAlmostDuplicate(vector<int> &nums, int indexDiff,
                                     int valueDiff) {
    set<int> record;
    for (int i = 0; i < nums.size(); ++i) {
      if (record.lower_bound(nums[i] - valueDiff) != record.end() &&
          *record.lower_bound(nums[i] - valueDiff) <= nums[i] + valueDiff) {
        return true;
      }

      record.insert(nums[i]);
      if (record.size() > indexDiff) {
        record.erase(nums[i - indexDiff]);
      }
    }
    return false;
  }
};

TEST(ContainsDuplicateIIITest, Normal) {
  vector<int> nums = {1, 2, 3, 1};
  int indexDiff = 3;
  int valueDiff = 0;
  ASSERT_TRUE(
      Solution().containsNearbyAlmostDuplicate(nums, indexDiff, valueDiff));

  nums = {1, 5, 9, 1, 5, 9};
  indexDiff = 2;
  valueDiff = 3;
  ASSERT_FALSE(
      Solution().containsNearbyAlmostDuplicate(nums, indexDiff, valueDiff));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
