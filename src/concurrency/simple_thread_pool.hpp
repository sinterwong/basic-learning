/**
 * @file simple_thread_pool.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-05-19
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _CONCURRENCY_SIMPLE_THREAD_POOL_HPP
#define _CONCURRENCY_SIMPLE_THREAD_POOL_HPP
#include "threadsafe_queue_fg.hpp"
#include "utils.hpp"
#include <atomic>
#include <functional>
#include <thread>

namespace my_concurrency {

class simple_thread_pool {
private:
  std::atomic_bool done;
  threadsafe_queue_fg_done<std::function<void()>> work_queue;
  std::vector<std::thread> threads;
  join_threads joiner;

  void worker_thread() {
    while (!done) {
      std::function<void()> task;
      if (work_queue.try_pop(task)) {
        task();
      } else {
        std::this_thread::yield();
      }
    }
  }

public:
  simple_thread_pool() : done(false), joiner(threads) {
    unsigned const thread_count = std::thread::hardware_concurrency();
    try {
      for (unsigned i = 0; i < thread_count; ++i) {
        threads.push_back(
            std::thread(&simple_thread_pool::worker_thread, this));
      }
    } catch (...) {
      done = true;
      throw;
    }
  }

  ~simple_thread_pool() { done = true; }
  template <typename FunctionType> void submit(FunctionType f) {
    work_queue.push(std::function<void()>(f));
  }
};

} // namespace my_concurrency

#endif // _CONCURRENCY_SIMPLE_THREAD_POOL_HPP
