/**
 * @file parallel_partial_sum.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 并行部分求和
 * @version 0.1
 * @date 2023-05-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _CONCUNNENCY_PARALLEL_PARTIAL_SUM_HPP
#define _CONCUNNENCY_PARALLEL_PARTIAL_SUM_HPP
#include "utils.hpp"
#include <algorithm>
#include <future>
#include <iterator>
#include <numeric>
#include <vector>
namespace concurrency {

template <typename Iterator>
void parallel_partial_sum(Iterator first, Iterator last) {
  using value_type = typename Iterator::value_type;

  struct process_chunk {
    void operator()(Iterator begin, Iterator last,
                    std::future<value_type> *previous_end_value,
                    std::promise<value_type> *end_value) {
      try {
        Iterator end = last; // 数据段的末项位置
        ++end;               // 末项位置的下一项
        // 执行数据段的计算
        std::partial_sum(begin, end, begin);
        if (previous_end_value) { // 如果有上一个数据段
          // 获取上一数据段的末项值(此处阻塞等待上一个数据段的计算完成)
          value_type addend = previous_end_value->get();
          // 将上一数据段的末项值加到当前数据段的首项
          *last += addend;
          // 如果有下一段，将当前数据段的末项值传递给调用者
          if (end_value) { // 如果有下一段
            end_value->set_value(*last);
          }
          // 将上一数据段的末项值加到当前数据段除了末项的所有项
          std::for_each(begin, last,
                        [addend](value_type &item) { item += addend; });
        } else if (end_value) {
          // 第一段数据，而且有下一段，将当前数据段的末项值传递给调用者
          end_value->set_value(*last);
        }
      } catch (...) {
        if (end_value) {
          // 如果还存在下一段就抛出线程异常
          end_value->set_exception(std::current_exception());
        } else {
          // 如果出现异常，但是没有调用者，意味着这是主线程异常，直接抛出即可
          throw;
        }
      }
    }
  };

  unsigned long const length = std::distance(first, last);
  if (!length) {
    return;
  }
  unsigned long const min_per_thread = 25;
  unsigned long const max_threads =
      (length + min_per_thread - 1) / min_per_thread;
  unsigned long const hardware_threads = std::thread::hardware_concurrency();
  unsigned long const num_threads =
      std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);
  unsigned long const block_size = length / num_threads;
  using value_type = typename Iterator::value_type;
  std::vector<std::thread> threads(num_threads - 1);
  // 依次存储各个数据段的末项值
  std::vector<std::promise<value_type>> end_values(num_threads - 1);
  // 获取以上数据段的末项值
  std::vector<std::future<value_type>> previous_end_values;
  previous_end_values.reserve(num_threads - 1);
  // 创建线程
  join_threads joiner(threads);
  Iterator block_start = first; // 数据段的起始位置
  for (unsigned long i = 0; i < (num_threads - 1); ++i) {
    Iterator block_last = block_start;
    // 数据段的末项位置
    std::advance(block_last, block_size - 1);
    // 创建线程
    threads[i] =
        std::thread(process_chunk(), block_start, block_last,
                    (i != 0) ? &previous_end_values[i - 1] : 0, &end_values[i]);
    // 存储数据段的末项值
    block_start = block_last;
    ++block_start;
    previous_end_values.push_back(end_values[i].get_future());
  }
  // 获取最后一个数据段的末项位置
  Iterator final_element = block_start;
  std::advance(final_element, std::distance(block_start, last) - 1);
  // 执行最后一个数据段
  process_chunk()(block_start, final_element,
                  (num_threads > 1) ? &previous_end_values.back() : 0, 0);
}

template <typename Iterator>
void parallel_partial_sum_2(Iterator first, Iterator last) {
  using value_type = typename Iterator::value_type;
  struct process_element {
    void operator()(Iterator first, Iterator last,
                    std::vector<value_type> &buffer, unsigned i, barrier &b) {
      value_type &ith_element = *(first + i); // 当前线程要计算的项, 也就是第i项
      // 是否需要更新源数据, 用于交替更新源数据
      bool update_source = false;
      // 从第一项开始, 每次步长翻倍
      for (unsigned step = 0, stride = 1; stride <= i; ++step, stride *= 2) {
        // 源数据, 也就是第i项, 用于计算
        value_type const &source = (step % 2) ? buffer[i] : ith_element;
        // 目标数据, 也就是第i项, 用于存储计算结果
        value_type &dest = (step % 2) ? ith_element : buffer[i];
        // 要加的数, 也就是第i - stride项
        value_type const &addend =
            (step % 2) ? buffer[i - stride] : *(first + i - stride);
        // 计算结果
        dest = source + addend;
        // 更新源数据
        update_source = !(step % 2);
        // 等待所有线程到达
        b.wait();
      }
      // 如果需要更新源数据, 将计算结果更新到源数据
      if (update_source) {
        ith_element = buffer[i];
      }
      // 等待所有线程到达
      b.done_waiting();
    }
  };

  unsigned long const length = std::distance(first, last);
  if (!length) {
    return;
  }
  std::vector<value_type> buffer(length); // 用于存储每个线程计算的结果
  barrier b(length); // 用于同步线程, 保证所有线程同时开始
  std::vector<std::thread> threads(length - 1); // 创建线程
  join_threads joiner(threads);
  Iterator block_start = first;
  for (unsigned long i = 0; i < (length - 1); ++i) {
    threads[i] =
        std::thread(process_element(), first, last, buffer, i, std::ref(b));
  }
}

} // namespace concurrency
#endif