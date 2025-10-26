/**
 * @file 11_container_with_most_water.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief You are given an integer array height of length n. There are n
vertical lines drawn such that the two endpoints of the ith line are (i, 0) and
(i, height[i]).

Find two lines that together with the x-axis form a container, such that the
container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.
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
  int maxArea(vector<int> &height) {
    int i = 0;
    int j = height.size() - 1;
    int max = 0;

    while (i < j) {
      int x = j - i;

      int y;
      if (height[i] < height[j]) {
        y = height[i++];
      } else {
        y = height[j--];
      }

      int currentArea = x * y;
      if (currentArea > max) {
        max = currentArea;
      }
    }

    return max;
  }
};

TEST(MaxAreaTest, Normal) {
  vector<int> height = {1, 8, 6, 2, 5, 4, 8, 3, 7};
  ASSERT_EQ(Solution().maxArea(height), 49);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
