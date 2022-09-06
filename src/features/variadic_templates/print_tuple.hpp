#ifndef __FEATURES_PRINT_TUPLE_HPP_
#define __FEATURES_PRINT_TUPLE_HPP_

#include <iostream>
#include <ostream>
#include <stdexcept>
#include <tuple>

namespace features {
namespace variadic_templates {

template <int IDX, int MAX, typename... Args> struct PRINT_TUPLE {
  static void print(std::ostream &os, std::tuple<Args...> const &t) {
    os << std::get<IDX>(t) << (IDX + 1 == MAX ? "" : ",");
    PRINT_TUPLE<IDX + 1, MAX, Args...>::print(os, t);
  }
};

template <int MAX, typename... Args> struct PRINT_TUPLE<MAX, MAX, Args...> {
  static void print(std::ostream &os, std::tuple<Args...> const &t) {}
};

template <typename... Args>
std::ostream &operator<<(std::ostream &os, std::tuple<Args...> const &t) {
  os << "[";
  PRINT_TUPLE<0, sizeof...(Args), Args...>::print(os, t);
  return os << "]";
}

} // namespace variadic_templates
} // namespace features

#endif