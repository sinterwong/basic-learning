#ifndef _MY_LOGGER_HPP__
#define _MY_LOGGER_HPP__

#include <atomic>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>

// Forward declaration
class LogImpl;

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_BLUE "\x1b[34m"
#define ANSI_COLOR_RESET "\x1b[0m"

enum class LogLevel { INFO, WARNING, ERROR, FATAL };

class Logger {
private:
  class LogImpl;

public:
  struct LogConfig {
    std::string logPath;
    std::string appName = "App";
    LogLevel logLevel = LogLevel::INFO;
    bool enableConsole = true;
    bool enableColor = true;
  };

  static Logger *instance();
  void initialize(const LogConfig &config);
  void shutdown();

  const char *getColorPrefix(LogLevel severity) const;
  const char *getColorSuffix() const;

  bool isInitialized() const;
  const LogConfig &getConfig() const { return config_; }

private:
  Logger();
  ~Logger();

  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  std::mutex mutex_;
  std::atomic<bool> isInitialized_;
  LogConfig config_;
  std::unique_ptr<LogImpl> pimpl_;
};

class MyLogMessage {
public:
  MyLogMessage(const char *file, int line, LogLevel severity);
  ~MyLogMessage();

  std::ostream &stream(); // Will be implemented in logger.cpp

private:
  // google::LogMessage glog_message_; // This will be part of LogImpl or
  // handled differently
  LogLevel severity_;     // To store the log level
  class MyLogMessageImpl; // Forward declaration for Pimpl for MyLogMessage
  std::unique_ptr<MyLogMessageImpl> pimpl_; // Pimpl
};

#define LOG_STREAM(level)                                                      \
  MyLogMessage(__FILE__, __LINE__, LogLevel::level).stream()

#define LOG_INFOS LOG_STREAM(INFO)
#define LOG_WARNINGS LOG_STREAM(WARNING)
#define LOG_ERRORS LOG_STREAM(ERROR)
#define LOG_FATALS LOG_STREAM(FATAL)
#endif // _MY_LOGGER_HPP__
