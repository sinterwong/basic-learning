/**
 * @file other_merge_intervals.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an array of intervals where intervals[i] = [starti, endi], merge
 * all overlapping intervals, and return an array of the non-overlapping
 * intervals that cover all the intervals in the input.
 * @version 0.1
 * @date 2024-10-15
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  vector<vector<int>> mergeIntervals(vector<vector<int>> &intervals) {
    vector<vector<int>> mergedRets;
    if (intervals.empty()) {
      return mergedRets;
    }

    sort(intervals.begin(), intervals.end());
    mergedRets.push_back(intervals[0]);

    for (int i = 1; i < intervals.size(); ++i) {
      auto lastEnd = mergedRets.back().at(1);
      auto currentBegin = intervals[i][0];
      auto currentEnd = intervals[i][1];

      if (lastEnd >= currentBegin) {
        // 说明在范围中，需要被合并
        mergedRets.back().at(1) = currentEnd;
      } else {
        mergedRets.push_back(intervals[i]);
      }
    }
    return mergedRets;
  }
};

TEST(MergeIntervalsTest, Normal) {
  vector<vector<int>> intervals = {{0, 1}, {2, 3}, {1, 3}, {5, 6}, {6, 8}};
  auto ret = Solution().mergeIntervals(intervals);
  ASSERT_EQ(ret.size(), 2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
