/**
 * @file thread_pool.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-05-16
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __SIMPLE_THREAD_POOL_H_
#define __SIMPLE_THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
namespace my_concurrency {

class task {
public:
  task() = default;

  template <typename Func> task(Func &&f) : ptr_(new wrapper{std::move(f)}){};

  void operator()() { ptr_->call(); }

  task(task &&) noexcept = default;
  task &operator=(task &&) noexcept = default;

  task(const task &) = delete;
  task &operator=(const task &) = delete;

  bool valid() const { return ptr_ != nullptr; }

private:
  class wrapper_base {
  public:
    virtual void call() = 0;
    virtual ~wrapper_base() {}
  };

  template <typename Func> class wrapper : public wrapper_base {
  public:
    wrapper(Func &&func) : f_(std::move(func)) {}
    virtual void call() override { f_(); };

  private:
    Func f_;
  };

  std::unique_ptr<wrapper_base> ptr_;
};

class thread_pool {
public:
  explicit thread_pool(size_t max_queue_size = 1024)
      : max_queue_size_(max_queue_size), state_(State::STOPPED) {}

  ~thread_pool() { stop(); }

  enum class State { STOPPED, RUNNING, STOPPING };

  void start(size_t n) {
    if (state_.exchange(State::RUNNING) != State::STOPPED) {
      return;
    }

    threads_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      threads_.emplace_back(&thread_pool::worker, this);
    }
  }

  void stop() {
    State expected = State::RUNNING;
    if (!state_.compare_exchange_strong(expected, State::STOPPING)) {
      return;
    }

    {
      std::unique_lock<std::mutex> lock(mutex_);
      state_ = State::STOPPING;
    }
    not_empty_.notify_all();
    not_full_.notify_all();

    for (auto &thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }

    threads_.clear();

    std::queue<task> empty;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      std::swap(task_queue_, empty);
      state_ = State::STOPPED;
    }
  }

  template <typename F, typename... Args>
  auto submit(F &&f,
              Args &&...args) -> std::future<std::invoke_result_t<F, Args...>> {
    using return_type = std::invoke_result_t<F, Args...>;

    if (state_ != State::RUNNING) {
      throw std::runtime_error("ThreadPool is not running");
    }

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> result = task->get_future();

    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!not_full_.wait_for(lock, std::chrono::seconds(5), [this] {
            return state_ != State::RUNNING ||
                   task_queue_.size() < max_queue_size_;
          })) {
        throw std::runtime_error("Queue is full");
      }

      if (state_ != State::RUNNING) {
        throw std::runtime_error("ThreadPool is stopping");
      }

      task_queue_.emplace([task]() { (*task)(); });
    }
    not_empty_.notify_one();

    return result;
  }

  template <typename F>
  auto submit(F &&f) -> std::future<std::invoke_result_t<F>> {
    using return_type = std::invoke_result_t<F>;

    if (state_ != State::RUNNING) {
      throw std::runtime_error("ThreadPool is not running");
    }

    auto task =
        std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
    std::future<return_type> result = task->get_future();

    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!not_full_.wait_for(lock, std::chrono::seconds(5), [this] {
            return state_ != State::RUNNING ||
                   task_queue_.size() < max_queue_size_;
          })) {
        throw std::runtime_error("Queue is full");
      }

      if (state_ != State::RUNNING) {
        throw std::runtime_error("ThreadPool is stopping");
      }

      task_queue_.emplace([task]() { (*task)(); });
    }
    not_empty_.notify_one();

    return result;
  }

  State get_state() const { return state_; }

private:
  void worker() {
    while (true) {
      task current_task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] {
          return state_ != State::RUNNING || !task_queue_.empty();
        });

        if (state_ != State::RUNNING && task_queue_.empty()) {
          return;
        }

        current_task = std::move(task_queue_.front());
        task_queue_.pop();
      }
      not_full_.notify_one();

      try {
        if (current_task.valid()) {
          current_task();
        }
      } catch (const std::exception &e) {
        std::cerr << "Task execution failed: " << e.what() << std::endl;
      } catch (...) {
        std::cerr << "Task execution failed with unknown error" << std::endl;
      }
    }
  }

  const size_t max_queue_size_;
  std::atomic<State> state_;
  std::vector<std::thread> threads_;
  std::queue<task> task_queue_;
  std::mutex mutex_;
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
};
} // namespace my_concurrency
#endif