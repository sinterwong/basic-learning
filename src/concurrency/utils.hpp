#ifndef _BASIC_CONCUNNENCY_UTILS_HPP_
#define _BASIC_CONCUNNENCY_UTILS_HPP_

#include <thread>
#include <vector>

namespace concurrency {

class join_threads {
  std::vector<std::thread> &threads;

public:
  explicit join_threads(std::vector<std::thread> &threads_)
      : threads(threads_) {}
  ~join_threads() {
    for (unsigned long i = 0; i < threads.size(); ++i) {
      if (threads[i].joinable()) {
        threads[i].join();
      }
    }
  }
};

} // namespace concurrency

#endif
