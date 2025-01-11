/**
 * @file 215_kth_largest_element.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an integer array nums and an integer k, return the kth largest
element in the array.
Note that it is the kth largest element in the sorted order, not the kth
distinct element.
Can you solve it without sorting?
 * @version 0.1
 * @date 2024-10-17
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <gtest/gtest.h>
#include <utility>

using namespace std;

class Solution {
public:
  int findKthLargest(vector<int> &nums, int k) {
    int left = 0;
    int right = nums.size() - 1;

    while (true) {
      std::swap(nums[left], nums[(rand() % (right - left + 1) + left)]);

      int p = nums[left];
      int lt = left;      // [l...lt) < p
      int gt = right + 1; // [gt...r] > p

      int i = left + 1; // [lt...gt) == p
      while (i < gt) {
        if (nums[i] < p) {
          swap(nums[i++], nums[++lt]);
        } else if (nums[i] > p) {
          swap(nums[i], nums[--gt]);
        } else {
          i++;
        }
      }
      swap(nums[lt], nums[left]);

      int ki = nums.size() - k;
      if (ki >= lt && ki < gt) {
        return nums[lt];
      } else if (ki < lt) {
        right = lt - 1;
      } else {
        left = gt;
      }
    }
  }
};

TEST(KthLargestTest, Normal) {
  vector<int> nums = {3, 2, 1, 5, 6, 3, 4};
  int k = 2;
  auto ret = Solution().findKthLargest(nums, k);
  ASSERT_EQ(ret, 5);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
