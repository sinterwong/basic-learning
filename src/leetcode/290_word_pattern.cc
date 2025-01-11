/**
 * @file 290_word_pattern.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given a pattern and a string s, find if s follows the same pattern.

Here follow means a full match, such that there is a bijection between a letter
in pattern and a non-empty word in s. Specifically:

Each letter in pattern maps to exactly one unique word in s.
Each unique word in s maps to exactly one letter in pattern.
No two letters map to the same word, and no two words map to the same letter.
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
  bool wordPattern(string pattern, string s) {
    vector<string> words = split(s);
    if (words.size() != pattern.size()) {
      return false;
    }

    map<char, string> patternMap;
    map<string, char> inputMap;

    for (int i = 0; i < words.size(); ++i) {
      char p = pattern[i];
      string const &word = words[i];
      if (patternMap.find(p) == patternMap.end()) {
        patternMap[p] = word;
      } else {
        if (patternMap[p] != word) {
          return false;
        }
      }

      if (inputMap.find(word) == inputMap.end()) {
        inputMap[word] = p;
      } else {
        if (inputMap[word] != p) {
          return false;
        }
      }
    }
    return true;
  }

private:
  std::vector<std::string> split(const std::string &s) {
    std::stringstream ss(s);
    std::string word;
    std::vector<std::string> result;
    while (ss >> word) {
      result.push_back(word);
    }
    return result;
  }
};

TEST(WordPatternTest, Normal) {
  string pattern = "abba";
  string s = "dog cat cat dog";
  ASSERT_TRUE(Solution().wordPattern(pattern, s));

  pattern = "abba";
  s = "dog cat cat fish";
  ASSERT_FALSE(Solution().wordPattern(pattern, s));

  pattern = "aaaa";
  s = "dog cat cat dog";
  ASSERT_FALSE(Solution().wordPattern(pattern, s));

  pattern = "jquery";
  s = "jquery";
  ASSERT_FALSE(Solution().wordPattern(pattern, s));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
