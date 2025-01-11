#include "thread_safe_queue.hpp"
#include <gtest/gtest.h>
#include <numeric>

using namespace my_concurrency;

using namespace std::chrono_literals;

TEST(ThreadSafeQueueTest, BasicFunctionality) {
  ThreadSafeQueue<int> queue;
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.size(), 0);

  queue.push(1);
  queue.push(2);
  queue.push(3);

  EXPECT_FALSE(queue.empty());
  EXPECT_EQ(queue.size(), 3);

  EXPECT_EQ(queue.try_pop().value(), 1);
  EXPECT_EQ(queue.try_pop().value(), 2);
  EXPECT_EQ(queue.try_pop().value(), 3);

  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.size(), 0);

  queue.push(4);
  EXPECT_EQ(queue.wait_pop(), 4);

  queue.push(5);
  EXPECT_EQ(queue.wait_pop_for(1s).value(), 5);

  queue.clear();
  EXPECT_TRUE(queue.empty());
}

TEST(ThreadSafeQueueTest, EmptyQueueTryPop) {
  ThreadSafeQueue<int> queue;
  EXPECT_EQ(queue.try_pop(), std::nullopt);
}

TEST(ThreadSafeQueueTest, WaitPopForTimeout) {
  ThreadSafeQueue<int> queue;
  EXPECT_EQ(queue.wait_pop_for(100ms), std::nullopt);
}

TEST(ThreadSafeQueueTest, MultiThreadedPush) {
  ThreadSafeQueue<int> queue;
  std::vector<std::thread> threads;
  const int num_threads = 10;
  const int num_items_per_thread = 100;

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&queue, num_items_per_thread]() {
      for (int j = 0; j < num_items_per_thread; ++j) {
        queue.push(j);
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_EQ(queue.size(), num_threads * num_items_per_thread);
}

TEST(ThreadSafeQueueTest, MultiThreadedTryPop) {
  ThreadSafeQueue<int> queue;
  const int num_threads = 10;
  const int num_items = num_threads * 100;

  for (int i = 0; i < num_items; ++i) {
    queue.push(i);
  }

  std::vector<std::thread> threads;
  std::atomic<int> popped_count = 0;

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&queue, &popped_count]() {
      while (true) {
        auto value = queue.try_pop();
        if (value) {
          popped_count++;
        } else {
          if (queue.empty()) {
            break;
          }
        }
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_EQ(popped_count, num_items);
  EXPECT_TRUE(queue.empty());
}

TEST(ThreadSafeQueueTest, MultiThreadedProducerConsumer) {
  ThreadSafeQueue<int> queue;
  std::vector<std::thread> threads;
  const int num_producers = 5;
  const int num_consumers = 5;
  const int num_items_per_producer = 100;

  for (int i = 0; i < num_producers; ++i) {
    threads.emplace_back([&queue, num_items_per_producer]() {
      for (int j = 0; j < num_items_per_producer; ++j) {
        queue.push(j);
      }
    });
  }

  std::atomic<int> sum = 0;
  for (int i = 0; i < num_consumers; ++i) {
    threads.emplace_back(
        [&queue, &sum, num_items_per_producer, num_producers]() {
          int local_sum = 0;
          for (int j = 0; j < num_items_per_producer; j++) {
            local_sum += queue.wait_pop();
          }
          sum += local_sum;
        });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(sum, (num_items_per_producer * (num_items_per_producer - 1) / 2) *
                     num_producers);
}

TEST(ThreadSafeQueueTest, MultiThreadedWaitPop) {
  ThreadSafeQueue<int> queue;
  std::vector<std::thread> threads;
  const int num_threads = 10;
  const int num_items = num_threads;

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&queue]() { queue.wait_pop(); });
  }

  std::this_thread::sleep_for(100ms);

  for (int i = 0; i < num_items; ++i) {
    queue.push(i);
  }

  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(queue.empty());
}

TEST(ThreadSafeQueueTest, MultiThreadedWaitPopFor) {
  ThreadSafeQueue<int> queue;
  std::vector<std::thread> threads;
  const int num_threads = 5;
  std::atomic<int> success_count = 0;

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&queue, &success_count]() {
      if (queue.wait_pop_for(200ms)) {
        success_count++;
      }
    });
  }

  std::this_thread::sleep_for(100ms);
  queue.push(1);
  std::this_thread::sleep_for(300ms);

  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_EQ(success_count, 1);
  EXPECT_TRUE(queue.empty());
}

TEST(ThreadSafePriorityQueueTest, BasicFunctionality) {
  ThreadSafePriorityQueue<int> queue;
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.size(), 0);

  queue.push(3);
  queue.push(1);
  queue.push(4);
  queue.push(2);

  EXPECT_FALSE(queue.empty());
  EXPECT_EQ(queue.size(), 4);

  EXPECT_EQ(queue.try_pop().value(), 4);
  EXPECT_EQ(queue.try_pop().value(), 3);
  EXPECT_EQ(queue.try_pop().value(), 2);
  EXPECT_EQ(queue.try_pop().value(), 1);

  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.size(), 0);

  queue.push(5);
  EXPECT_EQ(queue.wait_pop(), 5);

  queue.push(6);
  EXPECT_EQ(queue.wait_pop_for(1s).value(), 6);

  queue.clear();
  EXPECT_TRUE(queue.empty());
}

TEST(ThreadSafePriorityQueueTest, EmptyQueueTryPop) {
  ThreadSafePriorityQueue<int> queue;
  EXPECT_EQ(queue.try_pop(), std::nullopt);
}

TEST(ThreadSafePriorityQueueTest, WaitPopForTimeout) {
  ThreadSafePriorityQueue<int> queue;
  EXPECT_EQ(queue.wait_pop_for(100ms), std::nullopt);
}

TEST(ThreadSafePriorityQueueTest, CustomCompare) {
  ThreadSafePriorityQueue<int, std::greater<int>> queue;
  queue.push(3);
  queue.push(1);
  queue.push(4);
  queue.push(2);
}

TEST(ThreadSafePriorityQueueTest, MultiThreadedPush) {
  ThreadSafePriorityQueue<int> queue;
  std::vector<std::thread> threads;
  const int num_threads = 10;
  const int num_items_per_thread = 100;

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&queue, num_items_per_thread]() {
      for (int j = 0; j < num_items_per_thread; ++j) {
        queue.push(j);
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_EQ(queue.size(), num_threads * num_items_per_thread);
}

TEST(ThreadSafePriorityQueueTest, MultiThreadedTryPop) {
  ThreadSafePriorityQueue<int> queue;
  const int num_threads = 10;
  const int num_items = num_threads * 100;

  for (int i = 0; i < num_items; ++i) {
    queue.push(i);
  }

  std::vector<std::thread> threads;
  std::atomic<int> popped_count = 0;

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&queue, &popped_count]() {
      while (true) {
        auto value = queue.try_pop();
        if (value) {
          popped_count++;
        } else {
          if (queue.empty())
            break;
        }
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_EQ(popped_count, num_items);
  EXPECT_TRUE(queue.empty());
}

TEST(ThreadSafePriorityQueueTest, MultiThreadedProducerConsumer) {
  ThreadSafePriorityQueue<int> queue;
  std::vector<std::thread> threads;
  const int num_producers = 5;
  const int num_consumers = 5;
  const int num_items_per_producer = 100;

  for (int i = 0; i < num_producers; ++i) {
    threads.emplace_back([&queue, num_items_per_producer]() {
      for (int j = 0; j < num_items_per_producer; ++j) {
        queue.push(j);
      }
    });
  }

  std::vector<int> consumed_values;
  std::mutex consumed_mutex;

  for (int i = 0; i < num_consumers; ++i) {
    threads.emplace_back([&queue, &consumed_values, &consumed_mutex,
                          num_items_per_producer, num_producers]() {
      for (int j = 0; j < num_items_per_producer; j++) {
        int val = queue.wait_pop();
        std::lock_guard<std::mutex> lock(consumed_mutex);
        consumed_values.push_back(val);
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(queue.empty());
  std::sort(consumed_values.begin(), consumed_values.end());
  std::vector<int> expected_values(num_producers * num_items_per_producer);
  std::iota(expected_values.begin(), expected_values.end(), 0);
}

TEST(ThreadSafePriorityQueueTest, MultiThreadedWaitPop) {
  ThreadSafePriorityQueue<int> queue;
  std::vector<std::thread> threads;
  const int num_threads = 10;
  const int num_items = num_threads;

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&queue]() { queue.wait_pop(); });
  }

  std::this_thread::sleep_for(100ms);

  for (int i = 0; i < num_items; ++i) {
    queue.push(i);
  }

  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(queue.empty());
}

TEST(ThreadSafePriorityQueueTest, MultiThreadedWaitPopFor) {
  ThreadSafePriorityQueue<int> queue;
  std::vector<std::thread> threads;
  const int num_threads = 5;
  std::atomic<int> success_count = 0;

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&queue, &success_count]() {
      if (queue.wait_pop_for(200ms)) {
        success_count++;
      }
    });
  }

  std::this_thread::sleep_for(100ms);
  queue.push(1);
  std::this_thread::sleep_for(300ms);

  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_EQ(success_count, 1);
  EXPECT_TRUE(queue.empty());
}