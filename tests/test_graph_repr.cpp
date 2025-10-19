#include "graph/adj_matrix.hpp"
#include "graph/graph.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <logger.hpp>

namespace fs = std::filesystem;

class GraphRepresentationTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    Logger::LogConfig logConfig;
    logConfig.appName = "Graph-RepresentationTest-Unit-Test";
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

TEST_F(GraphRepresentationTest, AdjMatrix) {
  fs::path graphFile = dataDir / "chapter02" / "1_g.txt";
  bl::graph::AdjMatrix adjMatrix(graphFile.string());

  std::cout << adjMatrix << std::endl;
}

TEST_F(GraphRepresentationTest, AdjSet) {
  fs::path graphFile = dataDir / "chapter02" / "1_g.txt";
  bl::graph::Graph adjSet(graphFile.string());

  std::cout << adjSet << std::endl;
}
