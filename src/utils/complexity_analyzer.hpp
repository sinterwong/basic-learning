/**
 * @file complexity_analyzer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-10-16
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __UTILS_COMPLEXITY_ANALYZER_HPP_
#define __UTILS_COMPLEXITY_ANALYZER_HPP_

#include <cassert>
#include <chrono>
#include <cmath>
#include <vector>

namespace utils {
enum class TimeComplexity { O1, OLogN, ON, ONLogN, ON2, ON3, O2N, UNKNOWN };

// Overload the << operator for easy printing
inline std::ostream &operator<<(std::ostream &os, const TimeComplexity &tc) {
  switch (tc) {
  case TimeComplexity::O1:
    os << "O(1)";
    break;
  case TimeComplexity::OLogN:
    os << "O(logN)";
    break;
  case TimeComplexity::ON:
    os << "O(N)";
    break;
  case TimeComplexity::ONLogN:
    os << "O(NlogN)";
    break;
  case TimeComplexity::ON2:
    os << "O(N^2)";
    break;
  case TimeComplexity::ON3:
    os << "O(N^3)";
    break;
  case TimeComplexity::O2N:
    os << "O(2^N)";
    break;
  default:
    os << "UNKNOWN";
    break;
  }
  return os;
}

template <typename initFunc, typename F, typename... Args>
inline TimeComplexity analyzeTimeComplexity(int x, int lp, int gp,
                                            initFunc &&initArr, F &&func,
                                            Args... args) {
  assert(lp < gp);
  std::vector<double> times;

  for (int i = lp; i <= gp; i++) {
    int n = std::pow(x, i);
    std::vector<int> v(n);
    initArr(v.begin(), v.end());

    auto start = std::chrono::high_resolution_clock::now();
    func(v.begin(), v.end(), std::forward<Args>(args)...);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    times.push_back(static_cast<double>(duration));
  }

  // Analyze time complexity based on time ratios. This is a simplified
  // analysis. More rigorous analysis would require statistical methods and
  // curve fitting.
  if (times.size() >= 2) {
    double ratio1 = times[1] / times[0];
    double ratio2 = times.back() / times[times.size() - 2];

    if (std::abs(ratio1 - 1) < 0.1 && std::abs(ratio2 - 1) < 0.1)
      return TimeComplexity::O1;
    if (std::abs(ratio1 - x) < 0.5 * x && std::abs(ratio2 - x) < 0.5 * x)
      return TimeComplexity::ON;
    if (std::abs(ratio1 - x * std::log2(x)) < x &&
        std::abs(ratio2 - x * std::log2(x)) < x)
      return TimeComplexity::ONLogN;
    if (std::abs(ratio1 - x * x) < x * x && std::abs(ratio2 - x * x) < x * x)
      return TimeComplexity::ON2;
    if (std::abs(ratio1 - x * x * x) < x * x * x &&
        std::abs(ratio2 - x * x * x) < x * x * x)
      return TimeComplexity::ON3;
  }

  return TimeComplexity::UNKNOWN;
}

} // namespace utils
#endif
