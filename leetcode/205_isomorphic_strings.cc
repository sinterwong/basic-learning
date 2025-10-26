/**
 * @file 205_isomorphic_strings.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given two strings s and t, determine if they are isomorphic.

Two strings s and t are isomorphic if the characters in s can be replaced to get
t.

All occurrences of a character must be replaced with another character while
preserving the order of characters. No two characters may map to the same
character, but a character may map to itself.
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
  bool isIsomorphic(string s, string t) {
    if (s.size() != t.size()) {
      return false;
    }

    map<char, char> sMap;
    map<char, char> tMap;
    for (int i = 0; i < s.size(); ++i) {
      char sic = s[i];
      char tic = t[i];

      if (sMap.find(sic) != sMap.end()) {
        if (tic != sMap[sic]) {
          return false;
        }
      } else {
        sMap[sic] = tic;
      }

      if (tMap.find(tic) != tMap.end()) {
        if (sic != tMap[tic]) {
          return false;
        }
      } else {
        tMap[tic] = sic;
      }
    }
    return true;
  }
};

TEST(IsomorphicStringTest, Normal) {
  string s = "egg";
  string t = "add";
  ASSERT_TRUE(Solution().isIsomorphic(s, t));

  s = "foo";
  t = "bar";
  ASSERT_FALSE(Solution().isIsomorphic(s, t));

  s = "paper";
  t = "title";
  ASSERT_TRUE(Solution().isIsomorphic(s, t));

  s = "badc";
  t = "baba";
  ASSERT_FALSE(Solution().isIsomorphic(s, t));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
