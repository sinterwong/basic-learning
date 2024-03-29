#ifndef __DESIGIN_PATTERN_RECTANGLE_HPP_
#define __DESIGIN_PATTERN_RECTANGLE_HPP_
#include "factory.hpp"
#include "shape.hpp"
namespace oop {
namespace factory {
class Rectangle : public Shape {
public:
  Rectangle(int height_, int width_, int *k, std::string const &temp)
      : height(height_), width(width_), Shape() {}
  // Rectangle() : Shape(){}
  ~Rectangle() {}

  virtual void print() override {
    std::cout << "I'm sub class rectangle" << std::endl;
  }

private:
  int height;
  int width;
};

// static int __typeModuleClass = ObjFactory::regCreateObjFunc(
//     "Rectangle", (void *)(&__createObjFunc<Rectangle, int, int>));

BasicLearningModuleRegister(Rectangle, int, int, int *, std::string const &);
} // namespace factory
} // namespace oop
#endif
