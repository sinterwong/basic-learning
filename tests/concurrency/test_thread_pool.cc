#include "thread_pool.hpp"
#include <gtest/gtest.h>
#include <thread>

using namespace my_concurrency;

TEST(ThreadPoolTest, StartAndStop) {
  thread_pool pool;
  EXPECT_EQ(pool.get_state(), thread_pool::State::STOPPED);
  pool.start(2);
  EXPECT_EQ(pool.get_state(), thread_pool::State::RUNNING);
  pool.stop();
  EXPECT_EQ(pool.get_state(), thread_pool::State::STOPPED);
}

TEST(ThreadPoolTest, SubmitTask) {
  thread_pool pool;
  pool.start(2);
  auto future = pool.submit([] { return 1 + 2; });
  EXPECT_EQ(future.get(), 3);
  pool.stop();
}

TEST(ThreadPoolTest, SubmitMultipleTasks) {
  thread_pool pool;
  pool.start(4);
  std::vector<std::future<int>> futures;
  for (int i = 0; i < 10; ++i) {
    futures.push_back(pool.submit([i] { return i * 2; }));
  }
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(futures[i].get(), i * 2);
  }
  pool.stop();
}

TEST(ThreadPoolTest, SubmitTaskWithArguments) {
  thread_pool pool;
  pool.start(2);
  auto future = pool.submit([](int a, int b) { return a + b; }, 2, 3);
  EXPECT_EQ(future.get(), 5);
  pool.stop();
}

TEST(ThreadPoolTest, TaskExecutionOrder) {
  thread_pool pool;
  pool.start(1);
  std::atomic<int> counter = 0;
  std::vector<std::future<void>> futures;
  for (int i = 0; i < 5; ++i) {
    futures.push_back(pool.submit([&counter, i] {
      while (counter.load() != i)
        ;
      counter++;
    }));
  }
  for (auto &future : futures) {
    future.wait();
  }
  EXPECT_EQ(counter.load(), 5);
  pool.stop();
}

TEST(ThreadPoolTest, StopWhenTasksAreRunning) {
  thread_pool pool;
  pool.start(2);
  std::atomic<bool> flag1 = false;
  std::atomic<bool> flag2 = false;
  pool.submit([&flag1] {
    while (!flag1.load()) {
    }
  });
  pool.submit([&flag2] {
    while (!flag2.load()) {
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::jthread stop_thread([&pool] { pool.stop(); });

  flag1.store(true);
  flag2.store(true);

  stop_thread.join();
}

TEST(ThreadPoolTest, ThrowExceptionWhenNotRunning) {
  thread_pool pool;
  EXPECT_THROW(pool.submit([] { return 1; }), std::runtime_error);
  pool.start(2);
  pool.stop();
  EXPECT_THROW(pool.submit([] { return 1; }), std::runtime_error);
}

TEST(ThreadPoolTest, TaskExceptionHandling) {
  thread_pool pool;
  pool.start(1);
  auto future = pool.submit([] { throw std::runtime_error("Test Exception"); });

  try {
    future.get();
  } catch (const std::runtime_error &e) {
    EXPECT_STREQ("Test Exception", e.what());
  } catch (...) {
    FAIL() << "Expected std::runtime_error";
  }
  pool.stop();
}

TEST(ThreadPoolTest, StartStopStart) {
  thread_pool pool;
  pool.start(2);
  pool.stop();
}