#ifndef _BASIC_CONCUNNENCY_THREADSAFE_LIST_HPP_
#define _BASIC_CONCUNNENCY_THREADSAFE_LIST_HPP_

#include <memory>
#include <mutex>
namespace my_concurrency {
template <typename T> class threadsafe_list {
  struct node {
    std::mutex m;
    std::shared_ptr<T> data;
    std::unique_ptr<node> next;
    node() : next() {}
    node(T const &value) : data(std::make_shared<T>(value)) {}
  };
  node head;

public:
  threadsafe_list() {}
  ~threadsafe_list() {
    remove_if([](node const &) { return true; });
  }

  threadsafe_list(threadsafe_list const &) = delete;
  threadsafe_list &operator=(threadsafe_list const &) = delete;
  void push_front(T const &value) {
    // 初始化新节点
    std::unique_ptr<node> new_node = std::unique_ptr<node>(value);

    std::lock_guard<std::mutex> lk(head.m);
    // 新节点接替之前的头节点的next
    new_node->next = std::move(head.next);
    // 新节点成为头节点的next
    head.next = std::move(new_node);
  }

  template <typename Function> void for_each(Function f) {
    node *current = &head;
    std::unique_lock<std::mutex> current_lk(current->m);
    while (node *const next = current->next.get()) { // 条件声明语句
      std::unique_lock<std::mutex> next_lk(next->m);
      // 一旦我们取得了下一个节点的互斥，上一个节点就可以解锁了
      current_lk.unlock();
      // 对数据执行用户函数
      f(*next->data);

      // 更新当前节点
      current = next;
      current_lk = std::move(next_lk);
    }
  }

  template <typename Predicate> std::shared_ptr<T> find_first_if(Predicate p) {
    node *current = &head;
    std::unique_lock<std::mutex> current_lk(current->m);
    while (node *const next = current->next.get()) {
      std::unique_lock<std::mutex> next_lk(next->m);
      current_lk.unlock();

      if (p(*next->data)) {
        return next->data;
      }

      // 更新当前节点
      current = next;
      current_lk = std::move(next_lk);
    }
  }

  template <typename Predicate> void remove_if(Predicate p) {
    node *current = &head;
    std::unique_lock<std::mutex> current_lk(current->m);
    while (node *const next = current->next.get()) {
      std::unique_lock next_lk(next->m);
      if (p(*next->data)) {
        // 删除当前节点
        std::unique_ptr<node> old_next = std::move(current->next);
        current->next = std::move(next->next);
        next_lk.unlock();
      } else {
        // 更新当前节点
        current_lk.unlock();
        current = next;
        current_lk = std::move(next_lk);
      }
    }
  }
};
} // namespace my_concurrency

#endif
