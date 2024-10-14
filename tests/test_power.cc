
#include <cassert>
#include <gtest/gtest.h>

double power(int x, int n) {
  if (n == 0)
    return 1;

  if (n < 0) {
    return 1.0 / power(x, -n);
  }

  auto a = power(x, n / 2);

  if (n % 2 == 1) {
    return x * a * a;
  }
  return a * a;
}

class PowerTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(PowerTest, Positive) {
  ASSERT_EQ(power(2, 0), 1);
  ASSERT_EQ(power(2, 1), 2);
  ASSERT_EQ(power(2, 2), 4);
  ASSERT_EQ(power(2, 3), 8);
  ASSERT_EQ(power(2, 4), 16);
  ASSERT_EQ(power(2, 5), 32);
  ASSERT_EQ(power(2, 6), 64);
  ASSERT_EQ(power(2, 7), 128);
  ASSERT_EQ(power(2, 8), 256);
  ASSERT_EQ(power(2, 9), 512);
  ASSERT_EQ(power(2, 10), 1024);
}

TEST_F(PowerTest, Negative) {
  ASSERT_EQ(power(2, -1), 0.5);
  ASSERT_EQ(power(2, -2), 0.25);
  ASSERT_EQ(power(2, -3), 0.125);
  ASSERT_EQ(power(2, -4), 0.0625);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
