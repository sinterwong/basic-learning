#include "features/recursive_print.hpp"
#include "features/recursize_func_call.hpp"
#include "features/recursive_inheritance.hpp"
#include "features/my_printf.hpp"
#include "features/print_tuple.hpp"
#include <cstddef>
#include <iostream>

using namespace features::variadic_templates;

int main(int argc, char **argv) {
  // std::cout << "CPP version: " << __cplusplus << std::endl;

  print("hello", 5.2, 5, 'a'); // test recursive_print

  CustomerHash a("hello", "hi", 97);

  myPrintf("{} hello to {}.\n", "A", 10);

  std::cout << std::make_tuple("hello", 4, 78, 62.1) << std::endl;

  tuple<int, float, std::string> t(666, 1.2, "hello");
  std::cout << t.head() << std::endl;
  std::cout << t.tail().head() << std::endl;
  std::cout << t.tail().tail().head() << std::endl;

  return 0;
}