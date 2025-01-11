#ifndef _BASIC_CONCUNNENCY_PARALLEL_FOR_EACH_HPP_
#define _BASIC_CONCUNNENCY_PARALLEL_FOR_EACH_HPP_

#include "utils.hpp"
#include <algorithm>
#include <future>
#include <iostream>
#include <iterator>
#include <thread>

namespace my_concurrency {

template <typename Iterator, typename Func>
void parallel_for_each(Iterator first, Iterator last, Func func) {
  unsigned long const length = std::distance(first, last);

  if (!length) {
    return;
  }
  unsigned long const min_per_threads = 25;
  unsigned long const max_threads =
      (length + min_per_threads - 1) / min_per_threads;
  unsigned long const hardware_threads = std::thread::hardware_concurrency();
  std::cout << "hardware threads: " << hardware_threads << std::endl;
  unsigned long const num_threads =
      std::min((hardware_threads != 0 ? hardware_threads : 2), max_threads);
  unsigned long const block_size = length / num_threads;

  // 构建futures
  std::vector<std::future<void>> futures(num_threads - 1);

  // 创建执行线程
  std::vector<std::thread> threads(num_threads - 1);

  // 线程管理
  join_threads joiner(threads);

  Iterator block_start = first;
  // 开始执行任务
  for (unsigned long i = 0; i < (num_threads - 1); ++i) {
    Iterator block_end;
    std::advance(block_end, block_size);
    // 创建任务
    std::packaged_task<void(void)> task{
        [=]() { std::for_each(block_start, block_end, func); }};
    futures[i] = task.get_future();
    threads[i] = std::thread(std::move(task));
    block_start = block_end;
  }

  // 手动完成最后一组任务
  std::for_each(block_start, last, func);
  for (auto &f : futures) {
    f.get(); // 这里可以获取异常
  }
}

template <typename Iterator, typename Func>
void parallel_for_each_async(Iterator first, Iterator last, Func func) {

  unsigned long const min_per_threads = 25;
  unsigned long const length = std::distance(first, last);

  if (!length) {
    return;
  }
  if (length < (2 * min_per_threads)) {
    std::for_each(first, last, func);
  } else {
    Iterator const mid_point = first + length / 2; // 中间点
    auto future = std::async(&parallel_for_each_async<Iterator, Func>, first,
                             mid_point, func);
    parallel_for_each_async(mid_point, last, func);
    future.get();
  }
}
} // namespace my_concurrency

#endif
