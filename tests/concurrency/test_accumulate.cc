#include "parallel_accumulate_exception_safe.hpp"
#include <gtest/gtest.h>
#include <iostream>

TEST(ParallelAccumulateTest, Normal) {
  std::vector<int> data = {1, 2, 3, 4};
  int sum = my_concurrency::parallel_accumulate_exception_safe2(data.begin(),
                                                                data.end(), 0);

  ASSERT_EQ(sum, 10);
  std::cout << "sum: " << sum << std::endl;

  std::vector<int> data2 = {8, 8, 8, 8};
  sum = my_concurrency::parallel_accumulate_exception_safe(data2.begin(),
                                                           data2.end(), sum);
  ASSERT_EQ(sum, 42);
}