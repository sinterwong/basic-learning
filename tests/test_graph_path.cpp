#include "graph/connected_path.hpp"
#include "graph/graph.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <logger.hpp>

namespace fs = std::filesystem;

class GraphCPathTest : public ::testing::Test {
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

TEST_F(GraphCPathTest, Normal) {
  fs::path graphFile = dataDir / "chapter03" / "0_g.txt";
  bl::graph::Graph adjSet(graphFile.string());

  bl::graph::ConnectedPath cp(adjSet);

  std::vector<int> ret = cp.path(0, 4);
  std::vector<int> expected = {1, 4};
  ASSERT_EQ(ret.size(), expected.size());
  for (size_t i = 0; i < ret.size(); ++i) {
    ASSERT_EQ(ret[i], expected[i]);
  }

  ret = cp.path(0, 5);
  ASSERT_TRUE(ret.empty());
}