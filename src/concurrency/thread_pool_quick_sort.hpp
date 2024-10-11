#ifndef _BASIC_CONCUNNENCY_THREAD_POOL_QUICK_SORT_HPP_
#define _BASIC_CONCUNNENCY_THREAD_POOL_QUICK_SORT_HPP_

#include "threadsafe_stack.hpp"

#include <algorithm>
#include <atomic>
#include <future>
#include <list>
#include <memory>
#include <thread>
#include <vector>

namespace my_concurrency {
// FIXME:
template <typename T> struct quick_sorter {

  // 分治数据段
  struct chunk_to_sort {
    std::list<T> data;
    std::promise<std::list<T>> promise;

    chunk_to_sort() = default;

    chunk_to_sort(const chunk_to_sort &) = delete;
    chunk_to_sort &operator=(const chunk_to_sort &) = delete;

    chunk_to_sort(chunk_to_sort &&other) // move constructor
        : data(std::move(other.data)), promise(std::move(other.promise)) {}

    chunk_to_sort &
    operator=(chunk_to_sort &&other) { // move assignment operator
      data = std::move(other.data);
      promise = std::move(other.promise);
      return *this;
    }
  };

  threadsafe_stack<chunk_to_sort> chunks;
  std::vector<std::thread> threads;
  unsigned const max_thread_count;
  std::atomic<bool> end_of_data;
  quick_sorter()
      : max_thread_count(std::thread::hardware_concurrency() - 1),
        end_of_data(false) {}
  ~quick_sorter() {
    end_of_data = true;
    for (unsigned i = 0; i < threads.size(); ++i) {
      threads[i].join();
    }
  }

  void try_sort_chunk() {
    std::shared_ptr<chunk_to_sort> chunk = chunks.pop();
    if (chunk) {
      sort_chunk(chunk);
    }
  }

  std::list<T> do_sort(std::list<T> &chunk_data) {
    if (chunk_data.empty()) {
      return chunk_data;
    }
    std::list<T> result;
    // 第一条数据为基准数据
    result.splice(result.begin(), chunk_data, chunk_data.begin());
    T const &partition_val = *result.begin();

    // partition后基准数据的位置
    typename std::list<T>::iterator divide_point =
        std::partition(chunk_data.begin(), chunk_data.end(),
                       [&](T const &val) { return val < partition_val; });
    chunk_to_sort new_lower_chunk;
    new_lower_chunk.data.splice(new_lower_chunk.data.end(), chunk_data,
                                chunk_data.begin(), divide_point);

    std::future<std::list<T>> new_lower = new_lower_chunk.promise.get_future();
    chunks.push(std::move(new_lower_chunk));
    if (threads.size() < max_thread_count) {
      threads.push_back(std::thread(&quick_sorter<T>::sort_thread, this));
    }
    std::list<T> new_higher(do_sort(chunk_data));
    result.splice(result.end(), new_higher);
    while (new_lower.wait_for(std::chrono::seconds(0)) !=
           std::future_status::ready) {
      try_sort_chunk();
    }
    result.splice(result.begin(), new_lower.get());
    return result;
  }

  void sort_chunk(std::shared_ptr<chunk_to_sort> const &chunk) {
    chunk->promise.set_value(do_sort(chunk->data));
  }

  void sort_thread() {
    while (!end_of_data) {
      try_sort_chunk();
      std::this_thread::yield();
    }
  }
};

template <typename T> std::list<T> parallel_quick_sort(std::list<T> input) {
  if (input.empty()) {
    return input;
  }
  quick_sorter<T> s;
  return s.do_sort(input);
}

} // namespace my_concurrency

#endif
