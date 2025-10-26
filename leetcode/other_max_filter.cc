/**
 * @file other_max_filter.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-11-29
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <deque>
#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  // vector<int> max_filter(vector<int> &nums, int m) {
  //   // 时间复杂度：O(NM)，空间复杂度：O(1)
  //   vector<int> ans;
  //   for (int i = m - 1; i < nums.size(); ++i) {
  //     int maxVal = std::numeric_limits<int>::min();
  //     for (int j = i; j >= i - m + 1; --j) {
  //       if (maxVal < nums[j]) {
  //         maxVal = nums[j];
  //       }
  //     }
  //     ans.push_back(maxVal);
  //   }
  //   return ans;
  // }

  vector<int> max_filter(vector<int> &nums, int m) {
    // 时间复杂度：O(N)，空间复杂度：O(M)
    vector<int> ans;

    // 存储下标，使得对首总是当前窗口的最大值下标
    deque<int> dq;

    // 处理第一个窗口
    for (int i = 0; i < m; i++) {
      // 保持队列单调递减
      // 如果新数比队尾大，则队尾元素永远不可能是最大值，直接丢弃
      while (!dq.empty() && nums[i] >= nums[dq.back()]) {
        dq.pop_back();
      }
      dq.push_back(i);
    }

    // 将第一个窗口的最大值加入结果
    ans.push_back(nums[dq.front()]);

    for (int i = m; i < nums.size(); ++i) {
      // 删除已经不在窗口内的元素
      if (!dq.empty() && dq.front() <= i - m) {
        dq.pop_front();
      }

      // 保持队列单调递减
      while (!dq.empty() && nums[i] >= nums[dq.back()]) {
        dq.pop_back();
      }

      dq.push_back(i);

      // 队首元素即为当前窗口最大值
      ans.push_back(nums[dq.front()]);
    }

    return ans;
  }
};

TEST(MaxFilterTest, Normal) {
  vector<int> nums = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
  int m = 3;
  auto ans = Solution().max_filter(nums, m);
  ASSERT_EQ(ans, vector<int>({5, 7, 9, 9, 9, 6, 8, 10}));

  nums = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  ans = Solution().max_filter(nums, m);
  ASSERT_EQ(ans, vector<int>({10, 9, 8, 7, 6, 5, 4, 3}));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
