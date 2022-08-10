#ifndef __OOP_BASIC_TYPE_SHAPE_HPP_
#define __OOP_BASIC_TYPE_SHAPE_HPP_

namespace oop::factory {

class IShape {
public:
  virtual ~IShape() = default;
  virtual void print() = 0;
};
} // namespace oop::factory

#endif