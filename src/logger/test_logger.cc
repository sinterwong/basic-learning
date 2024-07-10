#include "logger.hpp"

int main() {
  BasicLearningLoggerInit(true, true, true, true);
  BASIC_LOGGER_TRACE("hello logger, {}", 2020);
  BASIC_LOGGER_DEBUG("hello logger, {}", 2020);
  BASIC_LOGGER_INFO("hello logger, {}", 2020);
  BASIC_LOGGER_WARN("hello logger, {}", 2020);
  BASIC_LOGGER_ERROR("hello logger, {}", 2020);
  BASIC_LOGGER_CRITICAL("hello logger, {}", 2020);
  BasicLearningLoggerDrop();
  return 0;
}
