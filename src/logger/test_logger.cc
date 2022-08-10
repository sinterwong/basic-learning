#include "logger.hpp"
#include <gtest/gtest.h>

class LoggerTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST(LoggerTest, Normal) {
  BasicLearningLoggerInit(true, true, true, true);
  BASIC_LOGGER_TRACE("hello logger, {}", 2020);
  BASIC_LOGGER_DEBUG("hello logger, {}", 2020);
  BASIC_LOGGER_INFO("hello logger, {}", 2020);
  BASIC_LOGGER_WARN("hello logger, {}", 2020);
  BASIC_LOGGER_ERROR("hello logger, {}", 2020);
  BASIC_LOGGER_CRITICAL("hello logger, {}", 2020);
  BasicLearningLoggerDrop();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}