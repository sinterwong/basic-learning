#include "oop/type_safe_factory.hpp"
#include <gtest/gtest.h>

namespace test_type_safe_factory {
using namespace oop::factory;

class ShapeBase {
public:
  virtual ~ShapeBase() {}
  virtual void print() const = 0;
};

class Rectangle : public ShapeBase {
public:
  explicit Rectangle(const ConstructorParams &params)
      : height_(get_param<int>(params, "height")),
        width_(get_param<int>(params, "width")) {}

  void print() const override {
    std::cout << "Rectangle: height=" << height_ << ", width=" << width_
              << std::endl;
  }

private:
  int height_;
  int width_;
};

class Circle : public ShapeBase {
public:
  explicit Circle(const ConstructorParams &params)
      : radius_(get_param<double>(params, "radius")) {}

  void print() const override {
    std::cout << "Circle: radius=" << radius_ << std::endl;
  }

private:
  double radius_;
};

class ShapeRegistrar {
public:
  static ShapeRegistrar &getInstance() {
    static ShapeRegistrar instance;
    return instance;
  }

  ShapeRegistrar(const ShapeRegistrar &) = delete;
  ShapeRegistrar &operator=(const ShapeRegistrar &) = delete;
  ShapeRegistrar(ShapeRegistrar &&) = delete;
  ShapeRegistrar &operator=(ShapeRegistrar &&) = delete;

private:
  ShapeRegistrar() {
    Factory<ShapeBase>::instance().registerCreator(
        "Rectangle",
        [](const ConstructorParams &params) -> std::shared_ptr<ShapeBase> {
          return std::make_shared<Rectangle>(params);
        });

    Factory<ShapeBase>::instance().registerCreator(
        "Circle",
        [](const ConstructorParams &params) -> std::shared_ptr<ShapeBase> {
          return std::make_shared<Circle>(params);
        });
  }
};

class TypeSafeFactoryTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TypeSafeFactoryTest, Normal) {
  // register class
  ShapeRegistrar::getInstance();

  ConstructorParams params_rectangle = {{"height", 10}, {"width", 20}};
  std::shared_ptr<ShapeBase> rectangle =
      Factory<ShapeBase>::instance().create("Rectangle", params_rectangle);
  EXPECT_NE(rectangle, nullptr);

  ConstructorParams params_circle = {{"radius", 5.0}};
  std::shared_ptr<ShapeBase> circle =
      Factory<ShapeBase>::instance().create("Circle", params_circle);
  EXPECT_NE(circle, nullptr);
}
} // namespace test_type_safe_factory