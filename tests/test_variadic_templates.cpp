#include "features/recursive_print.hpp"
// #include "features/recursize_func_call.hpp"
#include "features/recursive_inheritance.hpp"
#include <iostream>

using namespace features;

int main(int argc, char **argv) {
  std::cout << "CPP version: " << __cplusplus << std::endl;

  // variadic_templates::print("hello", 5.2, 5, 'a'); // test recursive_print

  // variadic_templates::hash_val("hi", "hello", "world");

  variadic_templates::tuple<int, float, std::string> t(1, 0.5, "hello");
  return 0;
}