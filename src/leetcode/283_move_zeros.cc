/**
 * @file 283_move_zeros.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an integer array nums, move all 0's to the end of it while
 * maintaining the relative order of the non-zero elements.
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
  // void moveZeroes(vector<int> &nums) {
  //   int k = 0; // 非0值的索引维护，循环完成后，k后的所有值都为0
  //   for (int i = 0; i < nums.size(); ++i) {
  //     if (nums[i] != 0) {
  //       if (i != k) {
  //         swap(nums[i], nums[k++]);
  //       } else {
  //         k++;
  //       }
  //     }
  //   }
  // }

  void moveZeroes(vector<int> &nums) {
    int slow = 0; // 快慢索引
    for (int fast = 0; fast < nums.size(); ++fast) {
      if (nums[fast] != 0) {
        nums[slow++] = nums[fast];
      }
    }

    for (int i = slow; i < nums.size(); i++) {
      nums[i] = 0;
    }
  }
};

TEST(MoveZerosTest, Normal) {
  vector<int> nums = {0, 1, 0, 3, 12};
  Solution().moveZeroes(nums);
  ASSERT_EQ(nums, vector<int>({1, 3, 12, 0, 0}));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
