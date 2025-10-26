/**
 * @file 26_remove_duplicates.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an integer array nums sorted in non-decreasing order, remove the
duplicates in-place such that each unique element appears only once. The
relative order of the elements should be kept the same. Then return the number
of unique elements in nums.

Consider the number of unique elements of nums to be k, to get accepted, you
need to do the following things:

Change the array nums such that the first k elements of nums contain the unique
elements in the order they were present in nums initially. The remaining
elements of nums are not important as well as the size of nums. Return k
 * @version 0.1
 * @date 2024-10-14
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  int removeDuplicates(vector<int> &nums) {
    if (nums.empty()) {
      return 0;
    }

    // 慢索引用于维护不重复元素的索引位置，快索引用于遍历数组
    int slow = 0;
    for (int fast = 1; fast < nums.size(); ++fast) {
      if (nums[slow] != nums[fast]) {
        nums[++slow] = nums[fast];
      }
    }

    return slow + 1;
  }
};

TEST(RemoveDuplicatesTest, Normal) {
  vector<int> nums = {0, 0, 1, 1, 1, 2, 2, 3, 3, 4};
  int k = Solution().removeDuplicates(nums);
  ASSERT_EQ(k, 5);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
