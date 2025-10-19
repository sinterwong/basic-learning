#include "dummy/dummy.hpp"
#include <gtest/gtest.h>
#include <logger.hpp>

class DummyTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    Logger::LogConfig logConfig;
    logConfig.appName = "Dummy-Test-Unit";
    logConfig.logPath = "./logs";
    logConfig.logLevel = LogLevel::INFO;
    logConfig.enableConsole = true;
    logConfig.enableColor = true;
    Logger::instance()->initialize(logConfig);
  }

  static void TearDownTestSuite() { Logger::instance()->shutdown(); }
};

TEST_F(DummyTest, Normal) {
  bl::dummy::MyDummy dummy;
  dummy.doSomething();
}
