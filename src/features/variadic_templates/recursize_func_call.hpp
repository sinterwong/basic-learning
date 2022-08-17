/**
 * @file recursize_func_call.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 递归函数调用，根据参数类型进入对应的特化模板
 * @version 0.1
 * @date 2022-08-17
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <cstddef>
#include <functional>
#include <iostream>

namespace features {
namespace variadic_templates {

/**
 * @brief hash_val 泛化模板（可以接收任何类型的参数，因此匹配优先级最低）
 *
 * @tparam Types
 * @param args
 * @return size_t
 */
template <typename... Types> inline size_t hash_val(Types const &...args) {
  size_t seed = 0;
  hash_val(seed, args...);
  return seed;
}

/**
 * @brief hash_val 部分特化模板
 *
 * @tparam T
 * @tparam Types
 * @param seed
 * @param val
 * @param args
 */
template <typename T, typename... Types>
inline void hash_val(size_t &seed, T const &val, Types const &...args) {
  hash_combine(seed, val);
  hash_val(seed, args...);
}

/**
 * @brief hash_val 更大部分特化模板
 *
 * @tparam T
 * @param seed
 * @param val
 */
template <typename T> inline void hash_val(size_t &seed, T const &val) {
  hash_combine(seed, val);
}

/**
 * @brief hash_combine
 *
 * @tparam T
 * @param seed
 * @param val
 */
template <typename T> inline void hash_combine(size_t &seed, T const &val) {
  seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

} // namespace variadic_templates
} // namespace features
