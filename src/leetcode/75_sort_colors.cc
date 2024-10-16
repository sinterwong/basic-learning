/**
 * @file 75_sort_colors.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an array nums with n objects colored red, white, or blue, sort
them in-place so that objects of the same color are adjacent, with the colors in
the order red, white, and blue.
We will use the integers 0, 1, and 2 to represent the color red, white, and
blue, respectively.
You must solve this problem without using the library's sort function.

 * @version 0.1
 * @date 2024-10-16
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  // void sortColors(vector<int> &nums) {
  //   // 计数排序，通常用于数量有限个类别
  //   // 时间复杂度：O(n)
  //   // 空间复杂度：O(n)
  //   array<int, 3> counts = {0, 0, 0};
  //   for (int i = 0; i < nums.size(); ++i) {
  //     assert(nums[i] <= 2 && nums[i] >= 0);
  //     counts[nums[i]]++;
  //   }
  //   vector<int> rets;
  //   for (int i = 0; i < counts.size(); i++) {
  //     for (int j = 0; j < counts[i]; ++j) {
  //       rets.push_back(i);
  //     }
  //   }
  //   nums = rets;
  // }

  void sortColors(vector<int> &nums) {
    // 利用三路快排单次partition的思想
    // 时间复杂度：O(n)
    // 空间复杂度：O(1)
    int lt = -1;          // [0....lt]
    int gt = nums.size(); // [gt....r)

    // [0...lt] = 0 (lt....gt) = 1 [gt....n) = 2
    for (int i = 0; i < gt;) {
      if (nums[i] == 0) {
        swap(nums[++lt], nums[i++]);
      } else if (nums[i] == 1) {
        i++;
      } else {
        assert(nums[i] == 2);
        // 交换完成后还需要再看看当前i的位置如何归置，所以本次i更新
        swap(nums[--gt], nums[i]);
      }
    }
  }
};

TEST(SortColorsTest, Normal) {
  vector<int> nums = {2, 0, 2, 1, 1, 0};
  Solution().sortColors(nums);
  ASSERT_EQ(nums, vector<int>({0, 0, 1, 1, 2, 2}));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
