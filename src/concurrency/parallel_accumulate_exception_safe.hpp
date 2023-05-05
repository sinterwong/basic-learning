#ifndef _BASIC_CONCUNNENCY_PARALLEL_ACCUMULATE_EXCEPTION_SAFE_HPP_
#define _BASIC_CONCUNNENCY_PARALLEL_ACCUMULATE_EXCEPTION_SAFE_HPP_

#include <future>
#include <iterator>
#include <numeric>
#include <thread>
#include <vector>

namespace concurrency {
template <typename Iterator, typename T> struct accumulate_block {
  T operator()(Iterator first, Iterator last) {
    return std::accumulate(first, last, T());
  }
};

class join_threads {
  std::vector<std::thread> threads;

public:
  explicit join_threads(std::vector<std::thread> const &threads_)
      : threads(threads_) {}
  ~join_threads() {
    for (unsigned long i = 0; i < threads.size(); ++i) {
      if (threads[i].joinable()) {
        threads[i].join();
      }
    }
  }
};

template <typename Iterator, typename T>
T parallel_accumulate_exception_safe(Iterator first, Iterator last, T init) {

  // 数据总量
  unsigned long const length = std::distance(first, last);

  // 每个线程至少执行的任务数量
  unsigned long const min_per_thread = 25;

  unsigned long const max_threads =
      (length + min_per_thread - 1) / min_per_thread;
  unsigned long const hardware_threads = std::thread::hardware_concurrency();
  unsigned long const num_threads =
      std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);

  unsigned long const block_size = length / num_threads;
  // 此时如果单个线程执行异常就可以通过future捕获，而不是直接在子线程中调用std::terminate终止程序
  std::vector<std::future<T>> futures(num_threads - 1);
  std::vector<std::thread> threads(num_threads - 1);
  join_threads joiner(threads);
  Iterator block_start = first;
  for (unsigned long i = 0; i < (num_threads - 1); ++i) {
    Iterator block_end = block_start;
    std::advance(block_end, block_size);

    std::packaged_task<T(Iterator, Iterator)> task{
        accumulate_block<Iterator, T>()};
    futures[i] = task.get_future();
    threads[i] = std::thread(std::move(task), block_start, block_end);
    block_start = block_end;
  }
  T last_result = accumulate_block<Iterator, T>()(block_start, last);
  T result = init;
  for (unsigned long i = 0; i < (num_threads - 1); ++i) {
    result += futures[i].get();
  }
  result += last_result;
  return result;
}

template <typename Iterator, typename T>
T parallel_accumulate_exception_safe2(Iterator first, Iterator last, T init) {
  unsigned long const length = std::distance(first, last);
  unsigned long const max_chunk_size = 25;
  if (length <= max_chunk_size) {
    return std::accumulate(first, last, init);
  } else {
    Iterator mid_point = first;
    std::advance(mid_point, length / 2);
    std::future<T> first_half_result =
        std::async(parallel_accumulate_exception_safe2<Iterator, T>(
            first, mid_point, init));

    T second_half_result =
        parallel_accumulate_exception_safe2(mid_point, last, T());
    return first_half_result.get() + second_half_result;
  }
}

} // namespace concurrency

#endif
