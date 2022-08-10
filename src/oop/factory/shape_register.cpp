#include "shape_register.hpp"
#include "factory.hpp"
#include "rectangle.hpp"

namespace oop::factory {
ShapeRegistrar::ShapeRegistrar() {
  BasicLearningModuleRegister(Rectangle, int, int);
}
} // namespace oop::factory
