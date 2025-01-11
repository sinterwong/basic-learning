/**
 * @file 447_number_of_boomerangs.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief You are given n points in the plane that are all distinct, where
points[i] = [xi, yi]. A boomerang is a tuple of points (i, j, k) such that the
distance between i and j equals the distance between i and k (the order of the
tuple matters).

Return the number of boomerangs.
 * @version 0.1
 * @date 2024-10-23
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <cmath>
#include <gtest/gtest.h>
#include <unordered_map>

using namespace std;

class Solution {
public:
  // int numberOfBoomerangs(vector<vector<int>> &points) {
  //   // 暴力枚举法 O(n^3)
  //   int ret = 0;
  //   for (int i = 0; i < points.size(); ++i) {
  //     for (int j = 0; j < points.size(); ++j) {
  //       if (i == j)
  //         continue;
  //       for (int k = 0; k < points.size(); ++k) {
  //         if (i == k || k == j)
  //           continue;

  //         if (dis(points[i], points[j]) == dis(points[i], points[k])) {
  //           ret++;
  //         }
  //       }
  //     }
  //   }
  //   return ret;
  // }

  int numberOfBoomerangs(vector<vector<int>> &points) {
    // 哈希表 O(n ^ 2)
    int ret = 0;
    for (int i = 0; i < points.size(); ++i) {
      unordered_map<int, int> disToCount;
      for (int j = 0; j < points.size(); ++j) {
        if (j != i) {
          disToCount[dis(points[i], points[j])]++;
        }
      }

      for (auto const &p : disToCount) {
        if (p.second >= 2) {
          // 只需要从n个距离相等的点中取两个点来组成三元组，A(n, 2) = n * n - 1
          ret += (p.second * (p.second - 1));
        }
      }
    }
    return ret;
  }

private:
  int dis(vector<int> &p1, vector<int> &p2) {
    return pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2);
  }
};

TEST(NumberOfBoomerangsTest, Normal) {
  vector<vector<int>> points = {{0, 0}, {1, 0}, {2, 0}};
  ASSERT_EQ(Solution().numberOfBoomerangs(points), 2);

  points = {{1, 1}, {2, 2}, {3, 3}};
  ASSERT_EQ(Solution().numberOfBoomerangs(points), 2);

  points = {{1, 1}};
  ASSERT_EQ(Solution().numberOfBoomerangs(points), 0);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
