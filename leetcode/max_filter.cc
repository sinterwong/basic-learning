/**
 * @file max_filter.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-11-26
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <deque>
#include <gtest/gtest.h>
using namespace std;

class Solution {
public:
  // vector<int> max_filter(vector<int> &nums, int n) {
  //   vector<int> ans;
  //   for (int i = n - 1; i < nums.size(); ++i) {
  //     int maxVal = std::numeric_limits<int>::min();
  //     for (int j = i; j >= i - n + 1; --j) {
  //       if (maxVal < nums[j]) {
  //         maxVal = nums[j];
  //       }
  //     }
  //     ans.push_back(maxVal);
  //   }
  //   return ans;
  // }

#include <deque>
#include <vector>

  vector<int> max_filter(vector<int> &nums, int n) {
    vector<int> ans;
    deque<int> dq; // 存储下标,队首总是当前窗口最大值的下标

    // 处理第一个窗口
    for (int i = 0; i < n; i++) {
      // 保持队列单调递减
      // 如果新数比队尾大,则队尾元素永远不可能是最大值,可以删除
      while (!dq.empty() && nums[i] >= nums[dq.back()]) {
        dq.pop_back();
      }
      dq.push_back(i);
    }

    // 将第一个窗口的最大值加入结果
    ans.push_back(nums[dq.front()]);

    // 处理后续窗口
    for (int i = n; i < nums.size(); i++) {
      // 删除已经不在窗口内的元素
      if (!dq.empty() && dq.front() <= i - n) {
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
  vector<int> nums = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
  Solution s;
  vector<int> ans = s.max_filter(nums, 3);

  for (auto i : ans) //[1,4,4,4,2,2,4]
    cout << i << " ";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
