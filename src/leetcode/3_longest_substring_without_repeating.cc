/**
 * @file 3_longest_substring_without_repeating.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given a string s, find the length of the longest
substring without repeating characters.
 * @version 0.1
 * @date 2024-10-19
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  int lengthOfLongestSubstring(string s) {

    if (s.empty()) {
      return 0;
    }

    // 滑动窗口范围：[l....r]
    int l = 0;
    int r = -1;
    int maxLen = -1;

    array<int, 256> dict = {0};
    while (l < s.size()) {
      if (r + 1 < s.size() && dict[s[r + 1]] == 0) {
        dict[s[++r]]++;
      } else {
        dict[s[l++]]--;
      }
      maxLen = max(maxLen, r - l + 1);
    }

    return maxLen;
  }
};

TEST(LSWRTest, Normal) {
  string s = "abcabcbb";
  ASSERT_EQ(Solution().lengthOfLongestSubstring(s), 3);

  s = "bbbbb";
  ASSERT_EQ(Solution().lengthOfLongestSubstring(s), 1);

  s = "pwwkew";
  ASSERT_EQ(Solution().lengthOfLongestSubstring(s), 3);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
