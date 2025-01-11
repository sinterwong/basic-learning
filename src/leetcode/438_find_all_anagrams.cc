/**
 * @file 438_find_all_anagrams.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given two strings s and p, return an array of all the start indices of
p's anagrams in s. You may return the answer in any order.
 * @version 0.1
 * @date 2024-10-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <cassert>
#include <gtest/gtest.h>
#include <unordered_map>

using namespace std;

class Solution {
public:
  // vector<int> findAnagrams(string s, string p) {
  //   // 暴力解法（O(N * M))
  //   vector<int> ret;
  //   unordered_map<char, int> pMap;

  //   int m = s.size();
  //   int n = p.size();

  //   if (n > m) {
  //     return ret; // p longer than s, no anagrams possible
  //   }
  //   for (auto c : p) {
  //     pMap[c]++;
  //   }

  //   for (int i = 0; i <= m - n; ++i) {
  //     unordered_map<char, int> sMap;
  //     for (int j = 0; j < n; ++j) {
  //       sMap[s[i + j]]++;
  //     }

  //     if (pMap == sMap) {
  //       ret.push_back(i);
  //     }
  //   }

  //   return ret;
  // }

  // vector<int> findAnagrams(string s, string p) {
  //   // 滑动窗口 + map (O(M + N))
  //   vector<int> ret;
  //   int m = s.size();
  //   int n = p.size();
  //   if (n > m) {
  //     return ret; // p longer than s, no anagrams possible
  //   }

  //   unordered_map<char, int> pMap;
  //   for (auto c : p) {
  //     pMap[c]++;
  //   }

  //   int l = 0;
  //   int r = -1;
  //   unordered_map<char, int> sMap;
  //   while (l <= m - n) {
  //     if (r + 1 < m && r - l + 1 < n) {
  //       sMap[s[++r]]++;
  //     } else {
  //       if (r - l + 1 == n && sMap == pMap) {
  //         ret.push_back(l);
  //       }

  //       sMap[s[l]]--;
  //       if (sMap[s[l]] == 0)
  //         sMap.erase(s[l]);
  //       l++;
  //     }
  //   }
  //   return ret;
  // }

  vector<int> findAnagrams(string s, string p) {
    // 滑动窗口 + array (O(M + N))
    vector<int> ret;
    int m = s.size();
    int n = p.size();
    if (n > m) {
      return ret; // p longer than s, no anagrams possible
    }

    array<char, 26> pCount = {0};
    array<char, 26> sCount = {0};
    for (auto c : p) {
      pCount[c - 'a']++;
    }

    int l = 0;
    int r = -1;
    while (l <= m - n) {
      int curr_size = r - l + 1;
      if (r + 1 < m && curr_size < n) {
        sCount[s[++r] - 'a']++;
      } else {
        assert(curr_size == n);
        if (sCount == pCount) {
          ret.push_back(l);
        }
        sCount[s[l++] - 'a']--;
      }
    }
    return ret;
  }
};

TEST(FindAnagramsTest, Normal) {
  string s = "cbaebabacd";
  string p = "abc";
  ASSERT_EQ(Solution().findAnagrams(s, p), vector<int>({0, 6}));

  s = "abab";
  p = "ab";
  ASSERT_EQ(Solution().findAnagrams(s, p), vector<int>({0, 1, 2}));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
