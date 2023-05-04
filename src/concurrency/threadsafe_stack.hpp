#ifndef _BASIC_CONCUNNENCY_SAFE_STACK_HPP_
#define _BASIC_CONCUNNENCY_SAFE_STACK_HPP_
#include <exception>
#include <memory>
#include <mutex>
#include <stack>

namespace concurrency {
struct empty_stack : std::exception {
  const char *what() const throw();
};

template <typename T> class threadsafe_stack {
private:
  std::stack<T> data;
  mutable std::mutex m;

public:
  threadsafe_stack(){};
  threadsafe_stack(threadsafe_stack const &other) {
    std::lock_guard<std::mutex> lk(other.m);
    data = other.data;
  }

  threadsafe_stack &operator=(threadsafe_stack const &other) = delete;

  void push(T new_value) {
    std::lock_guard<std::mutex> lk(m);
    data.push(new_value);
  }

  std::shared_ptr<T> pop() {
    std::lock_guard<std::mutex> lk(m);
    if (data.empty()) {
      return std::shared_ptr<T>();
    }
    std::shared_ptr<T> const ret(std::make_shared<T>(std::move(data.top())));
    data.pop();
    return ret;
  }

  void pop(T &value) {
    std::lock_guard<std::mutex> lk(m);
    if (data.empty()) {
      throw empty_stack();
    }
    value = std::move(data.top());
    data.pop();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lk(m);
    return data.empty();
  }
};
} // namespace concurrency

#endif
