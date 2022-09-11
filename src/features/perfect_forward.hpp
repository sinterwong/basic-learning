#ifndef __FEATURES_PERFECT_FORWARD_HPP_
#define __FEATURES_PERFECT_FORWARD_HPP_

#include <iostream>

namespace features {
namespace forward {
inline void _print(int &&a) { std::cout << "rvalue a: " << a << std::endl; }

inline void _print(int &a) { std::cout << "lvalue a: " << a << std::endl; }

// template<class T, class = typename
// std::enable_if<std::is_integral<T>::value>::type> void func(T a)
template <class T> void func(T &&a) { _print(std::forward<T>(a)); }
} // namespace forward
} // namespace features
#endif