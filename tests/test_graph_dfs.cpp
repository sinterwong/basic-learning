#include "graph/dfs.hpp"
#include "graph/graph.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <logger.hpp>

namespace fs = std::filesystem;

class GraphDeepFirstSearchTest : public ::testing::Test {
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

TEST_F(GraphDeepFirstSearchTest, DFSPreOrder) {
  fs::path graphFile = dataDir / "chapter03" / "0_g.txt";
  bl::graph::Graph adjSet(graphFile.string());

  bl::graph::GraphDFS dfs(adjSet);

  // 遍历某个联通分量
  std::vector<int> ret = dfs(0);
  std::vector<int> expected = {0, 1, 3, 2, 6, 4};
  ASSERT_EQ(ret.size(), expected.size());
  for (size_t i = 0; i < ret.size(); ++i) {
    ASSERT_EQ(ret[i], expected[i]);
  }

  // 遍历整个图
  ret = dfs();
  expected = {0, 1, 3, 2, 6, 4, 5};
  ASSERT_EQ(ret.size(), expected.size());
  for (size_t i = 0; i < ret.size(); ++i) {
    ASSERT_EQ(ret[i], expected[i]);
  }
}

TEST_F(GraphDeepFirstSearchTest, DFSPostOrder) {
  fs::path graphFile = dataDir / "chapter03" / "0_g.txt";
  bl::graph::Graph adjSet(graphFile.string());

  bl::graph::GraphDFS dfs(adjSet);

  // 遍历某个联通分量
  std::vector<int> ret = dfs(0, false);
  std::vector<int> expected = {6, 2, 3, 4, 1, 0};
  ASSERT_EQ(ret.size(), expected.size());
  for (size_t i = 0; i < ret.size(); ++i) {
    ASSERT_EQ(ret[i], expected[i]);
  }

  // 遍历整个图
  ret = dfs(false);
  expected = {6, 2, 3, 4, 1, 0, 5};
  ASSERT_EQ(ret.size(), expected.size());
  for (size_t i = 0; i < ret.size(); ++i) {
    ASSERT_EQ(ret[i], expected[i]);
  }
}