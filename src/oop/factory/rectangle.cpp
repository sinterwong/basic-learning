#include "rectangle.hpp"
#include <iostream>

namespace oop::factory {
Rectangle::Rectangle(int height_, int width_)
    : height(height_), width(width_) {}

void Rectangle::print() { std::cout << "I'm rectangle" << std::endl; }

} // namespace oop::factory
