#ifndef _BASIC_CONCUNNENCY_QUEUE_HPP_
#define _BASIC_CONCUNNENCY_QUEUE_HPP_

#include <memory>
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
};
} // namespace concurrency

#endif
