#include "memory_model.hpp"
#include <gtest/gtest.h>

using namespace op_system;

class MemoryModelTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(MemoryModelTest, Normal) {
  mem::MyMemoryModel model;
  uint8_t data[1024] = {0};
  uint8_t data_write[1024] = {0};
  for (int i = 0; i < 1024; ++i) {
    data_write[i] = i;
  }

  model.write(data_write, 0, 1024);

  model.read(data, 0, 1024);

  for (int i = 0; i < 1024; ++i) {
    ASSERT_EQ(data[i], data_write[i]);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
