#include <iostream>

#ifndef __DESIGIN_PATTERN_SHAPE_HPP_
#define __DESIGIN_PATTERN_SHAPE_HPP_

class Shape {
public:
  Shape() {}
  virtual ~Shape() {}
  virtual void print() = 0;
};

#endif