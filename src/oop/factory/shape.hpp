#include <iostream>

#ifndef __DESIGIN_PATTERN_SHAPE_HPP_
#define __DESIGIN_PATTERN_SHAPE_HPP_
namespace oop {
namespace factory {
class Shape {
public:
  Shape() {}
  virtual ~Shape() {}
  virtual void print() = 0;
};

} // namespace factory
} // namespace design_pattern
#endif