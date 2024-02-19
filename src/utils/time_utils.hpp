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
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

namespace utils {

/**
 * @brief 执行时间衡量
 *
 * @tparam F
 * @tparam Args
 * @param func
 * @param args
 * @return long long
 */
template <typename F, typename... Args>
inline long long measureTime(F func, Args &&...args) {
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

/**
 * @brief
 *
 * @tparam F
 * @param func
 * @param x
 * @param lp 从几次幂开始
 * @param gp 到几次幂结束
 */
template <typename F>
inline void testTimeByDataScaling(F &&func, int x, int lp, int gp) {
  assert(lp < gp);
  for (int i = lp; i < gp; i++) {
    int n = std::pow(x, i);

    // empty vector
    std::vector<int> v;
    v.resize(n);

    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "data size " << x << "^" << i << " = " << n << "\t";
    std::cout << "Time cost: "
              << static_cast<double>(duration.count()) / 1000000 << "s"
              << std::endl;
  }
}
} // namespace utils
#endif