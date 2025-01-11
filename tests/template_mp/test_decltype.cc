#include <iostream>
#include <vector>

#include <gtest/gtest.h>

template <typename T1, typename T2> auto add(T1 a, T2 b) -> decltype(a + b) {
  return a + b;
}

TEST(TestDecltype, GetValueTypeByObj) {
  std::vector<int> vec;
  vec.push_back(1);
  vec.push_back(2);
  vec.push_back(3);

  typename std::remove_reference<decltype(vec)>::type::value_type elem;
  typename std::remove_reference<decltype(vec)>::type::iterator ielem;
  typename std::remove_reference<decltype(vec)>::type containerType;

  std::cout << "Element Type: " << typeid(elem).name() << std::endl;
  std::cout << "Iterator Type: " << typeid(ielem).name() << std::endl;
  std::cout << "Container Type: " << typeid(containerType).name() << std::endl;
}

TEST(TestDecltype, Add) {
  int x = 5;
  double y = 3.14;
  auto result = add(x, y);
  ASSERT_DOUBLE_EQ(result, 8.14);
}
