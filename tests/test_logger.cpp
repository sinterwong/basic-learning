#include "logger/logger.hpp"
#include <gtest/gtest.h>

class LoggerTest : public ::testing::Test {
protected:
  void SetUp() override {
    LoggerInit(true, true, true, true);
    LoggerSetLevel(0);
  }
  void TearDown() override { LoggerDrop(); }
};

TEST_F(LoggerTest, Normal) {
  LOGGER_TRACE("hello basic learning, {}", 2025);
  LOGGER_DEBUG("hello basic learning, {}", 2025);
  LOGGER_INFO("hello basic learning, {}", 2025);
  LOGGER_WARN("hello basic learning, {}", 2025);
  LOGGER_ERROR("hello basic learning, {}", 2025);
  LOGGER_CRITICAL("hello basic learning, {}", 2025);
}
