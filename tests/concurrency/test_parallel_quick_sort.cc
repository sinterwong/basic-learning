#include "async_quick_sort.hpp"
#include "thread_pool_quick_sort.hpp"
#include <gtest/gtest.h>

class QuickSortTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  std::list<int> data = {5, 7, 3, 4, 1, 9, 2, 8, 10, 6};
  std::list<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
};

TEST_F(QuickSortTest, ThreadPoolQuickSort) {

  auto ret = my_concurrency::parallel_quick_sort(data);
  ASSERT_EQ(ret, expected);
}

TEST_F(QuickSortTest, SequentialQuickSort) {
  auto ret = my_concurrency::sequential_quick_sort(data);
  ASSERT_EQ(ret, expected);
}

TEST_F(QuickSortTest, ParallelQuickSort) {
  auto ret = my_concurrency::async_parallel_quick_sort(data);
  ASSERT_EQ(ret, expected);
}
