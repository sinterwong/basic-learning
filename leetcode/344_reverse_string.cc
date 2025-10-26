/**
 * @file 344_reverse_string.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Write a function that reverses a string. The input string is given as
an array of characters s.

You must do this by modifying the input array in-place with O(1) extra memory.
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
  void reverseString(vector<char> &s) {
    int i = 0;
    int j = s.size() - 1;

    while (i < j) {
      swap(s[i], s[j]);
      i++;
      j--;
    }
  }
};

TEST(ReverseStringTest, Normal) {
  vector<char> s = {'h', 'e', 'l', 'l', 'o'};
  Solution().reverseString(s);
  ASSERT_EQ(s, vector<char>({'o', 'l', 'l', 'e', 'h'}));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
