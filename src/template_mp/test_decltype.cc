#include <iostream>
#include <vector>

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
  /**
   * std::remove_reference<decltype(vec)>::type
   * 将 const std::vector<int>& 转换为 std::vector<int>
   */
  // 值类型
  typename std::remove_reference<decltype(vec)>::type::value_type elem;

  // 迭代器类型
  typename std::remove_reference<decltype(vec)>::type::iterator ielem;

  // 容器类型
  typename std::remove_reference<decltype(vec)>::type containerType;

  std::cout << "Element Type: " << typeid(elem).name() << std::endl;
  std::cout << "Iterator Type: " << typeid(ielem).name() << std::endl;
  std::cout << "Container Type: " << typeid(containerType).name() << std::endl;
}

int main() {
  // 测试 add 函数
  int x = 5;
  double y = 3.14;
  auto result = add(x, y);
  std::cout << "Result of add(5, 3.14): " << result << std::endl;
  std::cout << "Type of result: " << typeid(result).name() << std::endl;

  // 测试 getValTypeByObj 函数
  std::vector<int> intVec = {1, 2, 3};
  getValTypeByObj(intVec);

  return 0;
}