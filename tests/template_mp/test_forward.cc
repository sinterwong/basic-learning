#include <gtest/gtest.h>
#include <iostream>

inline void _print(int &&a) { std::cout << "rvalue a: " << a << std::endl; }

inline void _print(int &a) { std::cout << "lvalue a: " << a << std::endl; }

// template<class T, class = typename
// std::enable_if<std::is_integral<T>::value>::type> void func(T a)
template <class T> void func(T &&a) { _print(std::forward<T>(a)); }

TEST(TestForward, TestForward) {
  int a = 10;
  func(a);
  func(std::move(a));
}