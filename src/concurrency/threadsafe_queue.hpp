#ifndef _BASIC_CONCUNNENCY_SAFE_QUEUE_HPP_
#define _BASIC_CONCUNNENCY_SAFE_QUEUE_HPP_
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <queue>

namespace my_concurrency {
template <typename T> class threadsafe_queue {
private:
  // 在进行pop时make_shared有可能会抛出异常，导致push后notify的线程没有完成工作也不会去通知其他的线程
  // std::queue<T> data_queue;
  // 改用shared_ptr存储后，复制shared_ptr不会抛出异常
  std::queue<std::shared_ptr<T>> data_queue;
  std::condition_variable cond;
  mutable std::mutex m;

public:
  threadsafe_queue(){};

  void push(T new_value) {
    // 此处还有并发构建数据的能力，抛出异常也不会存在内存泄露的情况，也不会通知pop线程
    std::shared_ptr<T> data(std::make_shared<T>(std::move(new_value)));
    std::lock_guard<std::mutex> lk(m);
    data_queue.push(data);
    cond.notify_one(); // 通知一个线程开始执行条件变量
  }

  void wait_and_pop(T &value) {
    std::unique_lock<std::mutex> lk(m);
    // 等待条件变量通知
    cond.wait(lk, [this]() { return !data_queue.empty(); });
    value = std::move(*data_queue.front());
    data_queue.pop();
  }

  std::shared_ptr<T> wait_and_pop() {
    std::unique_lock<std::mutex> lk(m);
    cond.wait(lk, [this]() { return !data_queue.empty(); });
    std::shared_ptr<T> res = data_queue.front();
    data_queue.pop();
    return res;
  }

  bool try_pop(T &value) {
    std::lock_guard<std::mutex> lk(m);
    if (data_queue.empty()) {
      return false;
    }
    value = std::move(*data_queue.front());
    data_queue.pop();
    return true;
  }

  std::shared_ptr<T> try_pop() {
    std::lock_guard<std::mutex> lk(m);
    if (data_queue.empty()) {
      return std::shared_ptr<T>();
    }
    std::shared_ptr<T> ret = data_queue.front();
    data_queue.pop();
    return ret;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lk(m);
    return data_queue.empty();
  }
};
} // namespace my_concurrency

#endif
