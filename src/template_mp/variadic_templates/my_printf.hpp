#ifndef __FEATURES_MY_PRINTF_HPP_
#define __FEATURES_MY_PRINTF_HPP_

#include <iostream>
#include <stdexcept>

namespace template_mp {
namespace variadic_templates {

inline void myPrintf(char const *s) {
  // 递归到最后一次的调用
  while (*s) {
    if (*s == '{' && *(++s) == '}') {
      std::runtime_error("invalid format string: missing arguments!");
    }
    std::cout << *s++;
  }
}

template <typename T, typename... Args>
void myPrintf(char const *s, T const &value, Args const &...args) {

  while (*s) {
    if (*s == '{' && *(++s) == '}') {
      std::cout << value;
      myPrintf(++s, args...);
      return;
    }
    std::cout << *s++;
  }
  throw std::logic_error("extra arguments provided to printf");
}

} // namespace variadic_templates
} // namespace template_mp

#endif