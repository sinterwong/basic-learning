#include "logger.hpp"
#include "gtest/gtest.h"

class LoggerTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    Logger::LogConfig logConfig;
    logConfig.appName = "Test-Unit";
    logConfig.logPath = "./logs";
    logConfig.logLevel = LogLevel::INFO;
    logConfig.enableConsole = true;
    logConfig.enableColor = true;
    Logger::instance()->initialize(logConfig);
  }

  static void TearDownTestSuite() {
    // 程序退出前，手动关闭触发glog的shutdown来将缓冲区内容写入文件
    Logger::instance()->shutdown();
  }
};

TEST_F(LoggerTest, Normal) {
  LOG_INFOS << "(INFO) Hello World!";
  LOG_WARNINGS << "(WARN) Hello World!";
  LOG_ERRORS << "(ERROR) Hello World!";
  std::this_thread::sleep_for(std::chrono::seconds(1));
}

TEST_F(LoggerTest, AnotherTest) { LOG_INFOS << "This is another test case."; }
