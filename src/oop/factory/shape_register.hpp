#ifndef __OOP_BASIC_TYPE_SHAPE_REGISTER_HPP_
#define __OOP_BASIC_TYPE_SHAPE_REGISTER_HPP_

namespace oop::factory {

class ShapeRegistrar {
public:
  static ShapeRegistrar &getInstance() {
    static ShapeRegistrar instance;
    return instance;
  }

  ShapeRegistrar(const ShapeRegistrar &) = delete;
  ShapeRegistrar &operator=(const ShapeRegistrar &) = delete;

private:
  ShapeRegistrar();
};
} // namespace oop::factory

#endif