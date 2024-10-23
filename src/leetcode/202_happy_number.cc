/**
 * @file 202_happy_number.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Write an algorithm to determine if a number n is happy.

A happy number is a number defined by the following process:

Starting with any positive integer, replace the number by the sum of the squares
of its digits. Repeat the process until the number equals 1 (where it will
stay), or it loops endlessly in a cycle which does not include 1. Those numbers
for which this process ends in 1 are happy. Return true if n is a happy number,
and false if not.
 * @version 0.1
 * @date 2024-10-21
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <cmath>
#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  bool isHappy(int n) {
    set<int> alreadyExists;
    while (n != 1) {
      if (alreadyExists.find(n) != alreadyExists.end()) {
        return false;
      }

      alreadyExists.insert(n);
      n = squareSum(n);
    }

    return true;
  }

private:
  int squareSum(int n) {
    int sum = 0;
    for (; n > 0; n /= 10) {
      int k = n % 10;
      sum += k * k;
    }
    return sum;
  }
};

TEST(HappyNumberTest, Normal) {
  int n = 19;
  ASSERT_TRUE(Solution().isHappy(n));

  n = 2;
  ASSERT_FALSE(Solution().isHappy(n));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
