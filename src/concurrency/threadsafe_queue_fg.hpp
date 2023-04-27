#ifndef _BASIC_CONCUNNENCY_THREADSAFE_QUEUE_FG_HPP_
#define _BASIC_CONCUNNENCY_THREADSAFE_QUEUE_FG_HPP_

#include <condition_variable>
#include <memory>
#include <mutex>
namespace concurrency {
template <typename T> class queue {
private:
  struct node {
    T data;
    std::unique_ptr<node> next;
    node(T data_) : data(std::move(data_)) {}
  };
  std::unique_ptr<node> head; // 头节点
  node *tail;                 // 尾结点

public:
  queue() : tail(nullptr){};
  queue(const queue &other) = delete;
  queue &operator=(const queue &other) = delete;

  void push(T new_value) {
    std::unique_ptr<node> p = std::make_unique<node>(std::move(new_value));
    // 转移tail指针
    node *const new_tail = p.get();
    if (tail) {
      // 此前已经有数据入队
      tail->next = std::move(p); // 向后插入新节点
    } else {
      // 先前没有数据
      head = std::move(p); // 指点该数据为头节点
    }
    tail = new_tail; // 更新尾部节点
  }

  std::shared_ptr<T> try_pop() {
    // 从头部获取
    if (!head) {
      return std::shared_ptr<T>();
    }
    std::shared_ptr<T> const ret = std::make_shared<T>(std::move(head->data));
    // 更新头部节点为原头部节点的next
    std::unique_ptr<node> const old_head = std::move(head);
    head = std::move(old_head->next);
    if (!head) {
      // 已经没有任何数据，需要维护tail
      tail = nullptr;
    }
    return ret;
  }
};

template <typename T> class queue_dummy {
private:
  struct node {
    std::shared_ptr<T> data; // 改用shared_ptr存储
    std::unique_ptr<node> node;
  };
  std::unique_ptr<node> head;
  node *tail;

public:
  queue_dummy();
  queue_dummy(const queue_dummy &other) = delete;
  queue_dummy &operator=(const queue_dummy &other) = delete;

  std::shared_ptr<T> try_pop() {
    if (head.get == tail) {
      return std::shared_ptr<T>();
    }
    std::shared_ptr<T> const res = head->data;
    std::unique_ptr<node> old_head = std::move(head);
    head = std::move(old_head->next);
    return res;
  }

  void push(T new_value) {
    std::shared_ptr<T> new_data = std::make_shared<T>(std::move(new_value));
    std::unique_ptr<node> p = std::make_unique<node>();
    tail->data = new_data;
    node *const new_tail = p.get();
    tail->next = std::move(p);
    // 尾部永远指向一个dummy node
    tail = new_tail;
  }
};

template <typename T> class threadsafe_queue_fg {
private:
  struct node {
    std::shared_ptr<T> data;
    std::unique_ptr<node> next;
  };

  std::mutex head_mutex;
  std::unique_ptr<node> head;
  std::mutex tail_mutex;
  node *tail;

  node *get_tail() {
    // 获取tail指针是需要确保tail不会被操作
    std::lock_guard<std::mutex> tail_lock(tail_mutex);
    return tail;
  }

  std::unique_ptr<node> pop_head() {
    std::lock_guard<std::mutex> head_lock(head_mutex);
    if (head.get() == get_tail()) {
      return nullptr;
    }
    std::unique_ptr<node> old_head = std::move(head);
    head = std::move(old_head->next);
    return old_head;
  }

public:
  threadsafe_queue_fg() : head(std::make_unique<node>()), tail(head.get()) {}
  threadsafe_queue_fg(threadsafe_queue_fg const &other) = delete;
  threadsafe_queue_fg &operator=(threadsafe_queue_fg const &other) = delete;

  void push(T new_value) {
    std::shared_ptr<T> new_data = std::make_shared<T>(new_value);
    std::unique_ptr<node> p = std::make_unique<node>();

    node *const new_tail = p.get();

    // 对尾部指针操作时保护
    std::lock_guard<std::mutex> tail_lock(tail_mutex);
    tail->data = new_data;
    tail->next = std::move(p);
    tail = new_tail;
  }

  std::shared_ptr<T> try_pop() {
    std::unique_ptr<node> old_head = pop_head();
    return old_head ? old_head->data : std::shared_ptr<T>();
  }
};

template <typename T> class threadsafe_queue_fg_done {
private:
  struct node {
    std::shared_ptr<T> data;
    std::unique_ptr<node> next;
  };

  std::mutex head_mutex;
  std::unique_ptr<node> head;
  std::mutex tail_mutex;
  node *tail;
  std::condition_variable data_cond;

  node *get_tail() {
    std::lock_guard<std::mutex> tail_lock(tail_mutex);
    return tail;
  }

  std::unique_ptr<node> pop_head() {
    std::unique_ptr<node> old_head = std::move(head);
    head = std::move(old_head->next);
    return old_head;
  }

  std::unique_lock<std::mutex> wait_for_data() {
    std::unique_lock<std::mutex> head_lock(head_mutex);
    // 通知后的解锁条件为队列中存在数据
    data_cond.wait(head_lock, [&]() { return head.get() != get_tail(); });
    return std::move(head_lock);
  }

  std::unique_ptr<node> wait_pop_head() {
    std::unique_lock<std::mutex> head_lock(wait_for_data());
    return pop_head();
  }

  std::unique_ptr<node> wait_pop_head(T &value) {
    std::unique_lock<std::mutex> head_lock(wait_pop_head());
    value = std::move(*head->data);
    return pop_head();
  }

  std::unique_ptr<node> try_pop_head() {
    std::lock_guard<std::mutex> lock_head(head_mutex);
    if (head.get() == get_tail()) {
      return std::unique_ptr<node>();
    }
    return pop_head();
  }

  std::unique_ptr<node> try_pop_head(T &value) {
    std::lock_guard<std::mutex> lock_head(head_mutex);
    if (head.get() == get_tail()) {
      return std::unique_ptr<node>();
    }
    value = std::move(*head->data);
    return pop_head();
  }

public:
  threadsafe_queue_fg_done()
      : head(std::make_unique<node>()), tail(head.get()) {}
  threadsafe_queue_fg_done(threadsafe_queue_fg_done const &other) = delete;
  threadsafe_queue_fg_done &
  operator=(threadsafe_queue_fg_done const &other) = delete;

  void push(T new_value) {
    std::shared_ptr<T> new_data = std::make_shared<T>(std::move(new_value));
    std::unique_ptr<node> p = std::make_unique<node>(); // new dummy node
    node *const new_tail = p.get();
    {
      std::lock_guard<std::mutex> tail_lock(tail_mutex);
      tail->data = new_data;
      tail->next = std::move(p);
      tail = new_tail;
    }
    data_cond.notify_one();
  }

  std::shared_ptr<T> try_pop() {
    std::unique_ptr<node> old_head = try_pop_head();
    return old_head ? old_head->data : std::shared_ptr<T>();
  }

  bool try_pop(T &value) {
    std::unique_ptr<node> old_head = try_pop_head(value);
    return old_head;
  }

  std::shared_ptr<T> wait_and_pop() {
    std::unique_ptr<node> const old_head = wait_pop_head();
    return old_head->data;
  }

  void wait_and_pop(T &value) {
    std::unique_ptr<node> const old_head = wait_pop_head(value);
  }

  bool empty() {
    std::lock_guard<std::mutex> lock_head(head_mutex);
    return head.get == get_tail();
  }
};

} // namespace concurrency

#endif
