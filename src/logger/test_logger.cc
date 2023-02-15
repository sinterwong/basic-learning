#include "logger.hpp"
#include "spdlog/logger.h"
#include "spdlog/spdlog.h"

int main() {
  BasicLearningLoggerInit(true, true, true, true);

  // TODO C++20 编译报错
  // BASIC_LOGGER_TRACE("hello logger, {}", 2020);
  // BASIC_LOGGER_DEBUG("hello logger, {}", 2020);
  // BASIC_LOGGER_INFO("hello logger, {}", 2020);
  // BASIC_LOGGER_WARN("hello logger, {}", 2020);
  // BASIC_LOGGER_ERROR("hello logger, {}", 2020);
  // BASIC_LOGGER_CRITICAL("hello logger, {}", 2020);

  BasicLearningLoggerDrop();

  return 0;
}
