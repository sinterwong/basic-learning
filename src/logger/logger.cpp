#include "logger.hpp"
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <vector>

void Logger::init(const bool with_color_console, const bool with_console,
                  const bool with_error, const bool with_trace) {
  std::vector<spdlog::sink_ptr> sinks;

  if (with_color_console) {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sink->set_level(spdlog::level::trace);
    sinks.push_back(sink);
  } else if (with_console) {
    auto sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
    sink->set_level(spdlog::level::trace);
    sinks.push_back(sink);
  }

  if (with_error) {
    auto with_error_logger_rotating =
        std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            LOGGER_LOGGER_ERROR_FILENAME, LOGGER_ROTATING_MAX_FILE_SIZE,
            LOGGER_ROTATING_MAX_FILE_NUM);
    with_error_logger_rotating->set_level(spdlog::level::err);
    sinks.push_back(with_error_logger_rotating);
  }

  if (with_trace) {
    auto with_trace_logger_rotating =
        std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            LOGGER_LOGGER_TRACE_FILENAME, LOGGER_ROTATING_MAX_FILE_SIZE,
            LOGGER_ROTATING_MAX_FILE_NUM);
    with_trace_logger_rotating->set_level(spdlog::level::trace);
    sinks.push_back(with_trace_logger_rotating);
  }

  auto combined_logger =
      std::make_shared<spdlog::logger>(LOGGER_NAME, begin(sinks), end(sinks));
  combined_logger->set_level(spdlog::level::trace);
  combined_logger->set_pattern(LOGGER_PATTERN);
  spdlog::register_logger(combined_logger);

  setFlushEvery(2);
}

void Logger::setLevel(const int level) {
  std::shared_ptr<spdlog::logger> logger_ptr = spdlog::get(LOGGER_NAME);
  if (!logger_ptr) {
    fprintf(stderr, "Failed to get logger, Please init logger firstly.\n");
    return;
  }
  logger_ptr->set_level(static_cast<spdlog::level::level_enum>(level));
}

void Logger::setPattern(const char *format) {
  std::shared_ptr<spdlog::logger> logger_ptr = spdlog::get(LOGGER_NAME);
  if (!logger_ptr) {
    fprintf(stderr, "Failed to get logger, Please init logger firstly.\n");
    return;
  }
  logger_ptr->set_pattern(format);
}

void Logger::setFlushEvery(const int interval) {
  spdlog::flush_every(std::chrono::seconds(interval));
}

void Logger::drop() { spdlog::drop_all(); }