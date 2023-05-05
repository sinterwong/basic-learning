#include "factory.hpp"
#include "rectangle.hpp"
#include "shape.hpp"
#include <iostream>
#include <memory>
#include <unordered_map>

using namespace oop::factory;

template <typename... Args> int func(std::string const &a, Args... args) {
  std::cout << a << std::endl;
  return 10;
}

#define MyTemp(x, ...) static int ret_##x = func(#x, __VA_ARGS__)

int main(int argc, char **argv) {

  std::unordered_map<std::string, std::shared_ptr<Shape>> moduleMap;
  moduleMap["Rectangle"] = ObjFactory::createObj<Shape>("Rectangle", 5, 5);
  // Shape *shape;
  // shape = ObjFactory::createObj<Shape>("Rectangle", 5, 5);

  std::shared_ptr<Shape> shape =
      ObjFactory::createObj<Shape>("Rectangle", 5, 5);

  if (shape != nullptr) {
    shape->print();
  } else {
    std::cout << "Rectangle is null" << std::endl;
  }

  return 0;
}