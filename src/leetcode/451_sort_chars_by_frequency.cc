/**
 * @file 451_sort_chars_by_frequency.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given a string s, sort it in decreasing order based on the frequency
of the characters. The frequency of a character is the number of times it
appears in the string.

Return the sorted string. If there are multiple answers, return any of them.
 * @version 0.1
 * @date 2024-10-21
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  // string frequencySort(string s) {
  //   // O(NlogN)
  //   map<char, int> charFrequency;
  //   for (auto c : s) {
  //     charFrequency[c]++;
  //   }

  //   vector<pair<int, int>> vec(charFrequency.begin(), charFrequency.end());
  //   sort(vec.begin(), vec.end(),
  //        [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
  //          return a.second > b.second;
  //        });
  //   string ret = "";
  //   for (int i = 0; i < vec.size(); ++i) {
  //     for (int j = 0; j < vec[i].second; ++j) {
  //       ret += vec[i].first;
  //     }
  //   }
  //   return ret;
  // }

  string frequencySort(string s) {
    // 使用桶排序优化，整体时间复杂度为O(N)
    vector<int> freq(256, 0);
    for (char c : s) {
      freq[c]++;
    }

    vector<pair<int, char>> freq_vec;
    for (int i = 0; i < 256; ++i) {
      if (freq[i] > 0) {
        freq_vec.push_back({freq[i], i});
      }
    }

    sort(freq_vec.rbegin(), freq_vec.rend());

    string result = "";
    for (auto &p : freq_vec) {
      for (int i = 0; i < p.first; ++i) {
        result += p.second;
      }
    }

    return result;
  }
};

TEST(SortCharactersByFrequencyTest, Normal) {
  string s = "tree";
  ASSERT_EQ(Solution().frequencySort(s), "eetr");

  s = "cccaaa";
  ASSERT_EQ(Solution().frequencySort(s), "cccaaa");

  s = "Aabb";
  ASSERT_EQ(Solution().frequencySort(s), "bbaA");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
