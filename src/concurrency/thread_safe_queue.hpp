#ifndef __THREAD_SAFE_QUEUE_HPP_
#define __THREAD_SAFE_QUEUE_HPP_

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

namespace my_concurrency {

template <typename T> class ThreadSafeQueue {
public:
  ThreadSafeQueue() = default;

  void push(T value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(value));
    cv_.notify_one();
  }

  std::optional<T> try_pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return std::nullopt;
    }

    T value = std::move(queue_.front());
    queue_.pop();
    return value;
  }

  std::optional<T> wait_pop_for(const std::chrono::milliseconds &timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
      return std::nullopt;
    }

    T value = std::move(queue_.front());
    queue_.pop();
    return value;
  }

  T wait_pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty(); });

    T value = std::move(queue_.front());
    queue_.pop();
    return value;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::queue<T>().swap(queue_);
  }

private:
  mutable std::mutex mutex_;
  std::queue<T> queue_;
  std::condition_variable cv_;
};

template <typename T, typename Compare = std::less<T>>
class ThreadSafePriorityQueue {
public:
  ThreadSafePriorityQueue() = default;

  void push(T value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(value));
    cv_.notify_one();
  }

  std::optional<T> try_pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return std::nullopt;
    }

    T value = std::move(queue_.top());
    queue_.pop();
    return value;
  }

  std::optional<T> wait_pop_for(const std::chrono::milliseconds &timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
      return std::nullopt;
    }

    T value = std::move(queue_.top());
    queue_.pop();
    return value;
  }

  T wait_pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty(); });

    T value = std::move(queue_.top());
    queue_.pop();
    return value;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::priority_queue<T, std::vector<T>, Compare>().swap(queue_);
  }

private:
  mutable std::mutex mutex_;
  std::priority_queue<T, std::vector<T>, Compare> queue_;
  std::condition_variable cv_;
};
} // namespace my_concurrency

#endif