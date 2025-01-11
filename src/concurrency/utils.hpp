#ifndef _BASIC_CONCUNNENCY_UTILS_HPP_
#define _BASIC_CONCUNNENCY_UTILS_HPP_

#include <atomic>
#include <thread>
#include <vector>

namespace my_concurrency {

class join_threads {
  std::vector<std::thread> &threads;

public:
  explicit join_threads(std::vector<std::thread> &threads_)
      : threads(threads_) {}
  ~join_threads() {
    for (unsigned long i = 0; i < threads.size(); ++i) {
      if (threads[i].joinable()) {
        threads[i].join();
      }
    }
  }
};

struct barrier {
  std::atomic<unsigned> count;      // 等待的线程数
  std::atomic<unsigned> spaces;     // 等待的空位数
  std::atomic<unsigned> generation; // 代数
  barrier(unsigned count_) : count(count_), spaces(count_), generation(0) {}

  void wait() { // 等待所有线程到达, 然后一起执行。保证所有线程同时开始
    // 获取当前代数，用于自旋锁
    unsigned const gen = generation.load();
    // 等待的线程数减一
    if (!--spaces) { // 如果等待的线程数减一后为0, 说明所有线程都已经到达
      // 重置等待的线程数
      spaces = count.load();
      // 代数加一
      ++generation;
    } else {
      // 如果等待的线程数减一后不为0, 说明还有线程没有到达, 需要等待
      while (generation.load() == gen) { // 等待代数变化（自旋锁）
        std::this_thread::yield();       // 让出CPU
      }
    }
  }

  void done_waiting() { // 通知所有线程已经到达
    --count;            // 等待的线程数减一
    if (!--spaces) { // 如果等待的线程数减一后为0, 说明所有线程都已经到达
      spaces = count.load(); // 重置等待的线程数
      ++generation;          // 代数加一
    }
  }
};

} // namespace my_concurrency

#endif
