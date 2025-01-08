#ifndef _BASIC_CORE_LOGGER_LOGGER_HPP_
#define _BASIC_CORE_LOGGER_LOGGER_HPP_

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <spdlog/spdlog.h>

#define LOGGER_NAME "basic"
#define LOGGER_LOGGER_ERROR_FILENAME "logs/basic_error.log"
#define LOGGER_LOGGER_TRACE_FILENAME "logs/basic_error.log"
#define LOGGER_PATTERN "[%Y-%m-%d %H:%M:%S.%e][%^%l%$][%t][%s:%#] %v"
#define LOGGER_ROTATING_MAX_FILE_SIZE (1024 * 1024)
#define LOGGER_ROTATING_MAX_FILE_NUM 5

#define CHECK_LOGGER_INIT()                                                    \
  {                                                                            \
    auto logger = spdlog::get(LOGGER_NAME);                                    \
    if (!logger) {                                                             \
      Logger::getInstance().init(true, true, true, true);                      \
      Logger::getInstance().setLevel(2);                                       \
    }                                                                          \
  }

#define LOGGER_TRACE(...)                                                      \
  {                                                                            \
    CHECK_LOGGER_INIT();                                                       \
    SPDLOG_LOGGER_TRACE(spdlog::get(LOGGER_NAME), __VA_ARGS__);                \
  }

#define LOGGER_DEBUG(...)                                                      \
  {                                                                            \
    CHECK_LOGGER_INIT();                                                       \
    SPDLOG_LOGGER_DEBUG(spdlog::get(LOGGER_NAME), __VA_ARGS__);                \
  }

#define LOGGER_INFO(...)                                                       \
  {                                                                            \
    CHECK_LOGGER_INIT();                                                       \
    SPDLOG_LOGGER_INFO(spdlog::get(LOGGER_NAME), __VA_ARGS__);                 \
  }

#define LOGGER_WARN(...)                                                       \
  {                                                                            \
    CHECK_LOGGER_INIT();                                                       \
    SPDLOG_LOGGER_WARN(spdlog::get(LOGGER_NAME), __VA_ARGS__);                 \
  }

#define LOGGER_ERROR(...)                                                      \
  {                                                                            \
    CHECK_LOGGER_INIT();                                                       \
    SPDLOG_LOGGER_ERROR(spdlog::get(LOGGER_NAME), __VA_ARGS__);                \
  }

#define LOGGER_CRITICAL(...)                                                   \
  {                                                                            \
    CHECK_LOGGER_INIT();                                                       \
    SPDLOG_LOGGER_CRITICAL(spdlog::get(LOGGER_NAME), __VA_ARGS__);             \
  }

class Logger {
public:
  static Logger &getInstance() {
    static Logger instance;
    return instance;
  }

  void init(const bool with_color_console, const bool with_console,
            const bool with_error, const bool with_trace);
  void setLevel(const int level);
  void setPattern(const char *format);
  void setFlushEvery(const int interval);
  void drop();

  // Delete copy constructor and assignment operator
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

private:
  Logger() = default; // Private constructor
  ~Logger() = default;
};

// C-style interface wrappers
#ifdef __cplusplus
extern "C" {
#endif

inline void LoggerInit(const bool with_color_console, const bool with_console,
                       const bool with_error, const bool with_trace) {
  Logger::getInstance().init(with_color_console, with_console, with_error,
                             with_trace);
}

inline void LoggerSetLevel(const int level) {
  Logger::getInstance().setLevel(level);
}

inline void LoggerSetPattern(const char *format) {
  Logger::getInstance().setPattern(format);
}

inline void LoggerSetFlushEvery(const int interval) {
  Logger::getInstance().setFlushEvery(interval);
}

inline void LoggerDrop() { Logger::getInstance().drop(); }

#ifdef __cplusplus
}
#endif
#endif