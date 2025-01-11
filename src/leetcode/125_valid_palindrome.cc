/**
 * @file 125_valid_palindrome.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief A phrase is a palindrome if, after converting all uppercase letters
into lowercase letters and removing all non-alphanumeric characters, it reads
the same forward and backward. Alphanumeric characters include letters and
numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

 * @version 0.1
 * @date 2024-10-18
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <cctype>
#include <cwctype>
#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  bool isPalindrome(string s) {
    if (s.empty()) {
      return true;
    }
    // 转换字符串，大写转小写，去掉标点符号和空格
    string newS;
    for (char c : s) {
      if (isalnum(c)) {
        newS += tolower(c);
      }
    }

    int i = 0;
    int j = newS.size() - 1;
    while (i < j) {
      if (newS[i] != newS[j]) {
        return false;
      }
      i++;
      j--;
    }

    return true;
  }
};

TEST(IsPalindromeTest, Normal) {
  string s = "A man, a plan, a canal: Panama";
  ASSERT_TRUE(Solution().isPalindrome(s));

  s = "race a car";
  ASSERT_FALSE(Solution().isPalindrome(s));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
