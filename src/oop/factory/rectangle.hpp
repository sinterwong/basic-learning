#ifndef __OOP_BASIC_TYPE_RECTANGLE_HPP_
#define __OOP_BASIC_TYPE_RECTANGLE_HPP_
#include "shape.hpp"

namespace oop::factory {
class Rectangle : public IShape {
public:
  Rectangle(int height_, int width_);
  ~Rectangle() {}
  virtual void print() override;

private:
  int height;
  int width;
};
} // namespace oop::factory
#endif