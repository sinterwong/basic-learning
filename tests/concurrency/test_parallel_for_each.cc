#include "parallel_for_each.hpp"
#include <gtest/gtest.h>

TEST(ParallelForEachTest, Normal) {
  std::vector<int> data = {1, 2, 3, 4};
  my_concurrency::parallel_for_each(data.begin(), data.end(),
                                    [](int &it) { it += 1; });
  ASSERT_EQ(data[0], 2);
  ASSERT_EQ(data[1], 3);
  ASSERT_EQ(data[2], 4);
  ASSERT_EQ(data[3], 5);
}