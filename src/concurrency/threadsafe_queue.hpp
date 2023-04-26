#ifndef _BASIC_CONCUNNENCY_SAFE_QUEUE_HPP_
#define _BASIC_CONCUNNENCY_SAFE_QUEUE_HPP_
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <queue>

namespace concurrency {
template <typename T> class threadsafe_queue {
private:
  std::queue<T> data;
  std::condition_variable cond;
  mutable std::mutex m;

public:
  threadsafe_queue(){};

  void push(T new_value) {
    std::lock_guard<std::mutex> lk(m);
    data.push(new_value);
    cond.notify_one(); // 通知一个线程开始执行条件变量
  }

  void wait_and_pop(T &value) {
    std::unique_lock<std::mutex> lk(m);
    // 等待条件变量通知
    cond.wait(lk, [this]() { return !data.empty(); });
    value = std::move(data.front());
    data.pop();
  }

  std::shared_ptr<T> wait_and_pop() {
    std::unique_lock<std::mutex> lk(m);
    cond.wait(lk, [this]() { return !data.empty(); });
    std::shared_ptr<T> res{std::make_shared<T>(std::move(data.front()))};
    data.pop();
    return res;
  }

  bool try_pop(T &value) {
    std::lock_guard<std::mutex> lk(m);
    if (data.empty()) {
      return false;
    }
    value = std::move(data.top());
    data.pop();
    return true;
  }

  std::shared_ptr<T> try_pop() {
    std::lock_guard<std::mutex> lk(m);
    if (data.empty()) {
      return std::shared_ptr<T>();
    }
    std::shared_ptr<T> const ret(std::make_shared<T>(std::move(data.front())));
    data.pop();
    return ret;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lk(m);
    return data.empty();
  }
};
} // namespace concurrency

#endif
