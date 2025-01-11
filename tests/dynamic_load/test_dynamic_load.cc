/**
 * @file test_load_so.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Dynamic loading a shared library at runtime
 * @version 0.1
 * @date 2024-01-15
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <dlfcn.h>
#include <gtest/gtest.h>

TEST(DynamicLoadTest, Normal) {
  // load the library and get a handle.
  void *handle = dlopen("lib/dynamic_load/libmyadd.so", RTLD_LAZY);

  ASSERT_TRUE(handle);

  // load function symbols
  using add_function = int (*)(int, int);
  add_function my_add = (add_function)dlsym(handle, "_Z6my_addii");
  // if (!my_add) {
  //   const char *error = dlerror();
  //   if (error) {
  //     std::cerr << "Error loading symbol: " << error << std::endl;
  //   }
  //   std::cout << "failed to load functoin symbols." << std::endl;
  // }

  ASSERT_NE(nullptr, my_add);
  ASSERT_EQ(8, my_add(5, 3));

  dlclose(handle);
}
