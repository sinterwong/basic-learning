/**
 * @file alias_template.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @version 0.1
 * @date 2022-08-18
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __FEATURES_USING_DECLTYPE_HPP_
#define __FEATURES_USING_DECLTYPE_HPP_

#include <iostream>
#include <iterator>
#include <memory>
#include <vector>

namespace features {
namespace using_decltype {

/**
 * @brief 配合auto完成后置返回值（后置返回值意义在于可以使用参数信息）
 *
 * @tparam T1
 * @tparam T2
 * @param a
 * @param b
 * @return decltype(a + b)
 */
template <typename T1, typename T2> auto add(T1 a, T2 b) -> decltype(a + b) {

  return a + b;
}

/**
 * @brief 获取类型信息，类似于typeof但是可以直接使用类型
 * 
 * @tparam T 
 * @param vec 
 */
template <typename T> void getValTypeByObj(std::vector<T> const &vec) {
  // typename decltype(vec)::value_type elem;
  // T a{};
  // elem.push_back(a);
}

} // namespace using_decltype
} // namespace features

#endif