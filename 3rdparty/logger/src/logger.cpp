#include "logger.hpp"
#include <filesystem>
#include <glog/logging.h>
#include <iostream>

google::LogSeverity ToGlogSeverity(LogLevel level) {
  switch (level) {
  case LogLevel::INFO:
    return google::INFO;
  case LogLevel::WARNING:
    return google::WARNING;
  case LogLevel::ERROR:
    return google::ERROR;
  case LogLevel::FATAL:
    return google::FATAL;
  default:
    return google::INFO;
  }
}

class Logger::LogImpl {
public:
  LogImpl() = default;
  ~LogImpl() = default;

  void initializeGlog(const Logger::LogConfig &config) {
    google::InitGoogleLogging(config.appName.c_str());
    google::SetLogFilenameExtension(".log");

    std::string logDirectory = config.logPath;
    if (!logDirectory.empty() && logDirectory.back() != '/' &&
        logDirectory.back() != '\\') {
      logDirectory += '/';
    }

    if (!logDirectory.empty()) {
      std::filesystem::create_directories(std::filesystem::path(logDirectory));

      std::string log_file_prefix = logDirectory + config.appName;

      // 只需要为最低的日志级别设置目标。
      // 更高级别的日志（WARNING, ERROR, FATAL）会自动写入同一个文件。
      // 实际哪些级别的日志会被记录，由 FLAGS_minloglevel 控制。
      google::SetLogDestination(google::INFO, log_file_prefix.c_str());
    }

    FLAGS_minloglevel = ToGlogSeverity(config.logLevel);

    if (config.enableConsole) {
      FLAGS_alsologtostderr = true;
    }
  }

  void shutdownGlog() {
    if (Logger::instance()->isInitialized()) {
      google::ShutdownGoogleLogging();
    }
  }
};

class MyLogMessage::MyLogMessageImpl {
public:
  MyLogMessageImpl(const char *file, int line, LogLevel severity,
                   const Logger::LogConfig &config)
      : glog_message_(file, line, ToGlogSeverity(severity)) {
    if (config.enableColor) {
      glog_message_.stream() << Logger::instance()->getColorPrefix(severity);
    }
  }

  ~MyLogMessageImpl() {
    if (Logger::instance()->getConfig().enableColor) {
      glog_message_.stream() << Logger::instance()->getColorSuffix();
    }
  }

  std::ostream &stream() { return glog_message_.stream(); }

private:
  google::LogMessage glog_message_;
};

Logger *Logger::instance() {
  static Logger instance;
  return &instance;
}

Logger::Logger() : isInitialized_(false), pimpl_(std::make_unique<LogImpl>()) {}

Logger::~Logger() { shutdown(); }

void Logger::initialize(const LogConfig &config) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (isInitialized_.load()) {
    LOG(INFO) << "Logger already initialized";
    return;
  }

  config_ = config; // Store config

  if (pimpl_) {
    pimpl_->initializeGlog(config_);
  }

  google::LogMessage(__FILE__, __LINE__, google::INFO).stream()
      << "Logger initialized successfully";

  isInitialized_.store(true);
}

void Logger::shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (isInitialized_.load()) {
    if (pimpl_) {
      pimpl_->shutdownGlog();
    }
    isInitialized_.store(false);
  }
}

MyLogMessage::MyLogMessage(const char *file, int line, LogLevel severity)
    : severity_(severity) {
  if (Logger::instance()->isInitialized()) {
    pimpl_ = std::make_unique<MyLogMessageImpl>(
        file, line, severity, Logger::instance()->getConfig());
  }
}

MyLogMessage::~MyLogMessage() {}

std::ostream &MyLogMessage::stream() {
  if (pimpl_) {
    return pimpl_->stream();
  }
  static std::ostream null_stream(nullptr);
  return null_stream;
}

const char *Logger::getColorPrefix(LogLevel severity) const {
  if (!config_.enableColor) {
    return "";
  }

  switch (severity) {
  case LogLevel::INFO:
    return ANSI_COLOR_GREEN;
  case LogLevel::WARNING:
    return ANSI_COLOR_YELLOW;
  case LogLevel::ERROR:
    return ANSI_COLOR_RED;
  case LogLevel::FATAL:
    return ANSI_COLOR_RED;
  default:
    return "";
  }
}

const char *Logger::getColorSuffix() const {
  return config_.enableColor ? ANSI_COLOR_RESET : "";
}

bool Logger::isInitialized() const { return isInitialized_.load(); }
