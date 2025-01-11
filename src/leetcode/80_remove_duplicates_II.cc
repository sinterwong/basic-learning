/**
 * @file 80_remove_duplicates_II.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an integer array nums sorted in non-decreasing order, remove
some duplicates in-place such that each unique element appears at most twice.
The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you
must instead have the result be placed in the first part of the array nums. More
formally, if there are k elements after removing the duplicates, then the first
k elements of nums should hold the final result. It does not matter what you
leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the
input array in-place with O(1) extra memory.
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
  int removeDuplicates(std::vector<int> &nums) {
    if (nums.size() <= 2) {
      return nums.size();
    }

    // slow的定义为有效元素的个数。前两个元素一定有效
    int slow = 2;
    for (int fast = 2; fast < nums.size(); fast++) {
      // fast指向的元素不等于slow-2时，其出现的次数一定是小于2的，需要加入到有效元素中
      // 反之如果相等，那就证明其已经重复了三次（因为数组是有序的，所以1=3说明了1=2），因此是个无效值。
      if (nums[fast] != nums[slow - 2]) {
        nums[slow++] = nums[fast];
      }
    }
    return slow;
  }
};

TEST(RemoveDuplicatesIITest, Normal) {
  vector<int> nums = {1, 1, 1, 2, 2, 3};
  int k = Solution().removeDuplicates(nums);
  ASSERT_EQ(k, 5);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
