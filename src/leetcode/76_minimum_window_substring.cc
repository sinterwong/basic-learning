/**
 * @file 76_minimum_window_substring.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given two strings s and t of lengths m and n respectively, return the
minimum window substring of s such that every character in t (including
duplicates) is included in the window. If there is no such substring, return the
empty string "".

The testcases will be generated such that the answer is unique.
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
  // string minWindow(string s, string t) {
  //   // 暴力解法（O(N^3))
  //   int m = s.size();
  //   int n = t.size();

  //   if (m < n) {
  //     return "";
  //   }

  //   array<int, 256> tCount = {0};

  //   for (auto c : t) {
  //     tCount[c]++;
  //   }

  //   string minSubstring = "";
  //   for (int i = 0; i <= m - n; ++i) {
  //     for (int j = i; j < m; ++j) {
  //       if (j - i + 1 >= n) {
  //         array<int, 256> sCount = {0};
  //         for (int k = i; k <= j; ++k) {
  //           sCount[s[k]]++;
  //         }
  //         bool flag = true;
  //         for (int k = 0; k < 256; ++k) {
  //           if (sCount[k] < tCount[k]) {
  //             flag = false;
  //             break;
  //           }
  //         }
  //         if (flag &&
  //             (minSubstring.size() > (j - i + 1) || minSubstring.empty())) {
  //           minSubstring = string(s.begin() + i, s.begin() + j + 1);
  //         }
  //       }
  //     }
  //   }
  //   return minSubstring;
  // }

  // string minWindow(string s, string t) {
  //   // 滑动窗口（O(N))
  //   int m = s.size();
  //   int n = t.size();

  //   if (m < n) {
  //     return "";
  //   }

  //   array<int, 256> tCount = {0};
  //   array<int, 256> sCount = {0};

  //   for (auto c : t) {
  //     tCount[c]++;
  //   }

  //   string minSubstring = "";
  //   int l = 0;
  //   int r = -1;
  //   while (l <= m - n) {
  //     bool flag = isContain(sCount, tCount);
  //     if (r + 1 < m && !flag) {
  //       sCount[s[++r]]++;
  //     } else {
  //       if (flag) {
  //         if (minSubstring.empty() || minSubstring.size() > (r - l + 1)) {
  //           minSubstring = s.substr(l, r - l + 1);
  //         }
  //       }
  //       sCount[s[l++]]--;
  //     }
  //   }
  //   return minSubstring;
  // }

  string minWindow(string s, string t) {
    // 滑动窗口 + 内存优化
    int m = s.size();
    int n = t.size();

    if (m < n) {
      return "";
    }

    array<int, 256> tCount = {0};
    array<int, 256> sCount = {0};

    for (auto c : t) {
      tCount[c]++;
    }

    int minWindowLen = INT_MAX;
    int minWindowStart = 0;

    int l = 0;
    int r = -1;
    while (l <= m - n) {
      bool flag = isContain(sCount, tCount);
      if (r + 1 < m && !flag) {
        sCount[s[++r]]++;
      } else {
        if (flag) {
          if (r - l + 1 < minWindowLen) {
            minWindowLen = r - l + 1;
            minWindowStart = l;
          }
        }
        sCount[s[l++]]--;
      }
    }
    return minWindowLen == INT_MAX ? ""
                                   : s.substr(minWindowStart, minWindowLen);
  }

private:
  bool isContain(array<int, 256> const &sCount, array<int, 256> const &tCount) {
    for (int k = 0; k < 256; ++k) {
      if (sCount[k] < tCount[k]) {
        return false;
      }
    }
    return true;
  }
};

TEST(MinWindowSubstringTest, Normal) {
  string s = "ADOBECODEBANC";
  string p = "ABC";
  ASSERT_EQ(Solution().minWindow(s, p), "BANC");

  s = "a";
  p = "a";
  ASSERT_EQ(Solution().minWindow(s, p), "a");

  s = "a";
  p = "aa";
  ASSERT_EQ(Solution().minWindow(s, p), "");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
