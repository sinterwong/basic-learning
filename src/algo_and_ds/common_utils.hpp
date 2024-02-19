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

namespace algo_and_ds::utils {
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
 * @tparam Iter
 * @param first
 * @param last
 * @param func
 * @param x
 * @param lp
 * @param gp
 */
template <typename initFunc, typename F, typename... Args>
inline void testTimeByDataScaling(std::string des, int x, int lp, int gp,
                                  initFunc &&initArr, F &&func, Args... args) {
  assert(lp < gp);
  std::cout << "Test time by data scaling for (" << des << "): " << std::endl;
  for (int i = lp; i <= gp; i++) {
    int n = std::pow(x, i);

    // empty vector
    std::vector<int> v;
    v.resize(n);
    initArr(v.begin(), v.end());

    auto start = std::chrono::high_resolution_clock::now();
    func(v.begin(), v.end(), std::forward<Args>(args)...);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "data size " << x << "^" << i << " = " << n << "\t";
    std::cout << "Time cost: "
              << static_cast<double>(duration.count()) / 1000000 << "s"
              << std::endl;
  }
}
} // namespace algo_and_ds::utils
#endif