/**
 * @file 167_two_sum_II.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given a 1-indexed array of integers numbers that is already sorted in
non-decreasing order, find two numbers such that they add up to a specific
target number. Let these two numbers be numbers[index1] and numbers[index2]
where 1 <= index1 < index2 <= numbers.length.

Return the indices of the two numbers, index1 and index2, added by one as an
integer array [index1, index2] of length 2.

The tests are generated such that there is exactly one solution. You may not use
the same element twice.

Your solution must use only constant extra space.

 * @version 0.1
 * @date 2024-10-18
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  // vector<int> twoSum(vector<int> &numbers, int target) {
  //   // O(nlogn)
  //   for (int i = 0; i < numbers.size() - 1; ++i) {
  //     int e1 = numbers[i];

  //     int e2_i = binarySearch(numbers, i + 1, numbers.size(), target - e1);

  //     if (e2_i != -1) {
  //       return {i + 1, e2_i + 1};
  //     }
  //   }

  //   return {};
  // }

  vector<int> twoSum(vector<int> &numbers, int target) {
    // O(N)，对撞指针
    int i = 0;
    int j = numbers.size() - 1;
    while (numbers[i] + numbers[j] != target) {
      if (numbers[i] + numbers[j] < target) {
        i++;
      }

      if (numbers[i] + numbers[j] > target) {
        j--;
      }
    }

    return {i + 1, j + 1};
  }

private:
  int binarySearch(vector<int> &nums, int l, int r, int target) {
    if (l >= r) {
      return -1;
    }

    int mid = l + (r - l) / 2;
    if (nums[mid] == target) {
      return mid;
    }

    if (nums[mid] > target) {
      mid = binarySearch(nums, l, mid, target);
    } else {
      mid = binarySearch(nums, mid + 1, r, target);
    }
    return mid;
  }
};

TEST(TwoSumIITest, Normal) {
  vector<int> nums = {2, 7, 11, 15};
  int target = 9;
  vector<int> ret = Solution().twoSum(nums, target);
  ASSERT_EQ(ret, vector<int>({1, 2}));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
