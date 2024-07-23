#include "factory.hpp"
#include <iostream>
#include <memory>
#include <unordered_map>

using namespace oop::factory;

class Shape {
public:
  Shape() {}
  virtual ~Shape() {}
  virtual void print() = 0;
};

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

// register Rectangle
BasicLearningModuleRegister(Rectangle, int, int, int *, std::string const &);

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