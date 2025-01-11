/**
 * @file 345_reverse_vowels.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given a string s, reverse only all the vowels in the string and return
it.

The vowels are 'a', 'e', 'i', 'o', and 'u', and they can appear in both lower
and upper cases, more than once.
 * @version 0.1
 * @date 2024-10-18
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <gtest/gtest.h>
#include <unordered_set>

using namespace std;

class Solution {
public:
  string reverseVowels(string s) {
    unordered_set<char> dict = {'a', 'e', 'i', 'o', 'u',
                                'A', 'E', 'I', 'O', 'U'};
    int i = 0;
    int j = s.size() - 1;
    while (i < j) {
      while (i < j && dict.find(s[i]) == dict.end()) {
        i++;
      }

      while (i < j && dict.find(s[j]) == dict.end()) {
        j--;
      }

      if (i > j) {
        break;
      }
      swap(s[i++], s[j--]);
    }
    return s;
  }
};

TEST(ReverseStringTest, Normal) {
  string s = "IceCreAm";
  ASSERT_EQ(Solution().reverseVowels(s), "AceCreIm");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
