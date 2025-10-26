/**
 * @file 27_remove_element.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an integer array nums and an integer val, remove all occurrences
of val in nums in-place. The order of the elements may be changed. Then return
the number of elements in nums which are not equal to val.

Consider the number of elements in nums which are not equal to val be k, to get
accepted, you need to do the following things:

Change the array nums such that the first k elements of nums contain the
elements which are not equal to val. The remaining elements of nums are not
important as well as the size of nums. Return k.
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
  // int removeElement(vector<int> &nums, int val) {
  //   int k = 0; // 维护不等于k的数量
  //   for (int i = 0; i < nums.size(); i++) {
  //     if (nums[i] != val) {
  //       swap(nums[i], nums[k++]);
  //     }
  //   }
  //   return k;
  // }

  int removeElement(vector<int> &nums, int val) {
    int slow = 0; // 快慢索引
    for (int fast = 0; fast < nums.size(); fast++) {
      if (nums[fast] != val) {
        nums[slow++] = nums[fast];
      }
    }
    return slow;
  }
};

TEST(RemoveElementTest, Normal) {
  vector<int> nums = {3, 2, 2, 3};
  int val = 3;
  int k = Solution().removeElement(nums, val);
  ASSERT_EQ(k, 2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
