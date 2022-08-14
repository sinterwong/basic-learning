#include <iostream>
#include "desigin_pattern/reflection.h"
#include "desigin_pattern/shape.hpp"
#include "desigin_pattern/rectangle.hpp"

int main(int argc, char **argv) {

  Shape *shape;
  shape = ObjFactory::createObj<Shape>("Rectangle", 5, 5);

  if (shape != nullptr) {
    shape->print();
  } else {
    std::cout << "Rectangle is null" << std::endl;
  }

  return 0;
}