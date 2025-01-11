/**
 * @file 242_valid_anagram.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given two strings s and t, return true if t is an anagram of s, and
 * false otherwise.
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
  bool isAnagram(string s, string t) {
    if (s.size() != t.size()) {
      return false;
    }

    array<int, 26> sCount;
    for (auto c : s) {
      sCount[c - 'a']++;
    }

    for (auto c : t) {
      if (--sCount[c - 'a'] < 0) {
        return false;
      }
    }
    return true;
  }
};

TEST(ValidAnagramTest, Normal) {
  string s = "anagram";
  string t = "nagaram";
  ASSERT_TRUE(Solution().isAnagram(s, t));

  s = "rat";
  t = "car";
  ASSERT_FALSE(Solution().isAnagram(s, t));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
