#include "graph/cc.hpp"
#include "graph/graph.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <logger.hpp>

namespace fs = std::filesystem;

class GraphCCTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    Logger::LogConfig logConfig;
    logConfig.appName = "Graph-DFS-Unit-Test";
    logConfig.logPath = "./logs";
    logConfig.logLevel = LogLevel::INFO;
    logConfig.enableConsole = true;
    logConfig.enableColor = true;
    Logger::instance()->initialize(logConfig);
  }

  static void TearDownTestSuite() { Logger::instance()->shutdown(); }

  fs::path resourceDir = fs::path("assets");
  fs::path dataDir = resourceDir / "data" / "graph";
};

TEST_F(GraphCCTest, Normal) {
  fs::path graphFile = dataDir / "chapter03" / "0_g.txt";
  bl::graph::Graph adjSet(graphFile.string());

  bl::graph::ConnectedComponent cc(adjSet);

  ASSERT_EQ(cc.count(), 2);

  std::vector<std::vector<int32_t>> components = cc.components();
  ASSERT_EQ(components.size(), 2);
  ASSERT_EQ(components[0].size(), 6);
  ASSERT_EQ(components[1].size(), 1);

  ASSERT_TRUE(cc.isConnected(0, 6));
  ASSERT_FALSE(cc.isConnected(0, 5));
}