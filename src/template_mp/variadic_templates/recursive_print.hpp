#ifndef __FEATURES_RECURIVE_PRINT_HPP_
#define __FEATURES_RECURIVE_PRINT_HPP_

#include <iostream>

namespace template_mp {
namespace variadic_templates {

inline void print() {
  // 递归到最后会有一个无参的调用，这里需要适配一下
}

template <typename T, typename... Args>
void print(T const &firstArg, Args const &...args) {

  std::cout << firstArg << std::endl;
  std::cout << "number of args: " << sizeof...(args) << std::endl;
  // 每次递归忽略掉第一个参数，这样每次就会少一个（因为只有一个模板函数，没有特化）
  print(args...);
}

template <typename... Args> void print(Args const &...args) {
  std::cout << "hello world" << std::endl;
}

} // namespace variadic_templates
} // namespace template_mp

#endif