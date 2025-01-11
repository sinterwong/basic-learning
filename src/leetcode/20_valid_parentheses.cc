/**
 * @file 20_valid_parentheses.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given a string s containing just the characters '(', ')', '{', '}',
'[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Every close bracket has a corresponding open bracket of the same type.
 * @version 0.1
 * @date 2024-11-11
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <gtest/gtest.h>
#include <stack>

using namespace std;

class Solution {
public:
  bool isValid(string s) {
    stack<char> st;
    for (auto c : s) {
      if (c == '(' || c == '{' || c == '[') {
        st.push(c);
      } else {
        if (st.empty()) {
          return false;
        }
        auto tc = st.top();
        st.pop();
        if (tc == '(' && c != ')' || tc == '{' && c != '}' ||
            tc == '[' && c != ']') {
          return false;
        }
      }
    }
    if (!st.empty()) {
      return false;
    }
    return true;
  }
};

TEST(ValidParenthesesTest, Normal) {
  string s = "()";
  ASSERT_TRUE(Solution().isValid(s));

  s = "()[]{}";
  ASSERT_TRUE(Solution().isValid(s));

  s = "(]";
  ASSERT_FALSE(Solution().isValid(s));

  s = "([)]";
  ASSERT_FALSE(Solution().isValid(s));

  s = "{[]}";
  ASSERT_TRUE(Solution().isValid(s));

  s = "";
  ASSERT_TRUE(Solution().isValid(s));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
