#include "features/recursive_print.hpp"
#include "features/recursize_func_call.hpp"
#include "features/my_printf.hpp"
#include <cstddef>
#include <iostream>

using namespace features::variadic_templates;

int main(int argc, char **argv) {
  // std::cout << "CPP version: " << __cplusplus << std::endl;

  print("hello", 5.2, 5, 'a'); // test recursive_print

  CustomerHash a("hello", "hi", 97);

  myPrintf("{} hello to {}.\n", "A", 10);

  return 0;
}