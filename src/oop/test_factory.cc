#include "factory.hpp"
#include "shape.hpp"
#include "shape_register.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <unordered_map>

using namespace oop::factory;

class FactoryTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST(FactoryTest, Normal) {
  ShapeRegistrar::getInstance();

  std::unordered_map<std::string, std::shared_ptr<IShape>> moduleMap;
  moduleMap["Rectangle"] = ObjFactory::createObj<IShape>("Rectangle", 5, 5);

  auto rectanglePtr = moduleMap["Rectangle"];

  if (rectanglePtr != nullptr) {
    rectanglePtr->print();
  } else {
    std::cout << "rectanglePtr is null" << std::endl;
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}