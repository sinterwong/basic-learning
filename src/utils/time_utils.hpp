/**
 * @file time_utils.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-28
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __FLOWENGINE_TIME_UTILS_H_
#define __FLOWENGINE_TIME_UTILS_H_
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <thread>
#include <utility>

using namespace std::chrono_literals;

namespace utils {

/**
 * @brief 执行时间衡量
 *
 * @tparam F
 * @tparam Args
 * @param func
 * @param args
 * @return int64_t
 */
template <typename F, typename... Args>
inline int64_t measureTime(F func, Args &&...args) {
  auto start = std::chrono::high_resolution_clock::now();
  func(std::forward<Args>(args)...);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  return duration.count();
}

/**
 * @brief 每隔一定时间执行一次任务
 *
 * @tparam F
 * @tparam Args
 * @param interval
 * @param f
 * @param args
 */
template <typename F, typename... Args>
inline void periodicTask(std::chrono::milliseconds interval, F &&f,
                         Args &&...args) {
  auto lastExecTime = std::chrono::steady_clock::now();
  while (true) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - lastExecTime);
    if (elapsed >= interval) {
      std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
      lastExecTime = now;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
} // namespace utils
#endif