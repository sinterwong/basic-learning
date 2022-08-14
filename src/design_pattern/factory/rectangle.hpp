#ifndef __DESIGIN_PATTERN_RECTANGLE_HPP_
#define __DESIGIN_PATTERN_RECTANGLE_HPP_
#include "shape.hpp"
#include "reflection.h"

class Rectangle : public Shape {
public:
  // Rectangle(int height_, int width_)
  //     : height(height_), width(width_), Shape() {}
  Rectangle() : Shape(){}
  ~Rectangle() {}

  virtual void print() override {
    std::cout << "I'm sub class rectangle" << std::endl;
  }

private:
  int height;
  int width;
};

ModuleRegister(Rectangle);

#endif