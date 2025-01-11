/**
 * @file 149_max_points.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an array of points where points[i] = [xi, yi] represents a point
 * on the X-Y plane, return the maximum number of points that lie on the same
 * straight line.
 * @version 0.1
 * @date 2024-10-23
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <algorithm>
#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  int maxPoints(vector<vector<int>> &points) {
    // 暴力枚举解法 O(n ^ 3)
    int n = points.size();
    if (n <= 2) {
      return n;
    }
    int maxNums = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        int count = 0;
        for (int k = 0; k < n; ++k) {
          if (areCollinear(points[i], points[j], points[k])) {
            count++;
          }
        }
        maxNums = max(maxNums, count);
      }
    }
    return maxNums;
  }

private:
  bool areCollinear(const vector<int> &p1, const vector<int> &p2,
                    const vector<int> &p3) {
    /**
     * @brief 使用叉积判断三点共线
     * Δx_ik = points[k][0] - points[i][0]
     * Δy_ik = points[k][1] - points[i][1]
     * Δx_ij = points[j][0] - points[i][0]
     * Δy_ij = points[j][1] - points[i][1]
     *
     * 几何意义：
     * 假设点 i、j、k 不重合。
     * 如果点 i、j、k 共线，那么由点 i 和 j 构成的向量与由点 i 和 k
     * 构成的向量是线性相关的，这意味着这两个向量的斜率相等。
     * 斜率的计算公式是 Δy /Δx。
     * 所以，如果斜率相等，则有：(Δy_ij / Δx_ij) == (Δy_ik / Δx_ik)
     * 交叉相乘得：Δy_ij * Δx_ik == Δy_ik * Δx_ij（避免潜在的除零）
     */
    //
    return (p2[1] - p1[1]) * (p3[0] - p1[0]) ==
           (p3[1] - p1[1]) * (p2[0] - p1[0]);
  }
};

TEST(MaxPointsOnALineTest, Normal) {
  vector<vector<int>> points = {{1, 1}, {2, 2}, {3, 3}};
  ASSERT_EQ(Solution().maxPoints(points), 3);

  points = {{1, 1}, {3, 2}, {5, 3}, {4, 1}, {2, 3}, {1, 4}};
  ASSERT_EQ(Solution().maxPoints(points), 4);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
