#include <iostream>

namespace features {
namespace variadic_templates {

void print() {
  // 递归到最后会有一个无参的调用，这里需要适配一下
}

template <typename T, typename... Args>
void print(T const &firstArg, Args const &...args) {

  std::cout << firstArg << std::endl;
  std::cout << "number of args: " << sizeof...(args) << std::endl;
  // 每次递归忽略掉第一个参数，这样每次就会少一个（因为只有一个模板函数，没有特化）
  print(args...);
}

} // namespace variadic_templates
} // namespace features
