/**
 * @file test_bellman_ford_sp.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-05
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "bellman_ford_sp.hpp"
#include "weighted_read_graph.hpp"
#include "weighted_sparse_graph.hpp"
#include <filesystem>
#include <gtest/gtest.h>

using namespace algo_and_ds::graph;
namespace fs = std::filesystem;

class BellmanFordSPTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  fs::path dataDir = "assets/data/graph";
};

TEST_F(BellmanFordSPTest, Normal) {
  std::string f = (dataDir / "testG8.txt").string();
  int V = 5;
  SparseGraph<int> g = SparseGraph<int>(V, true);
  ReadGraph<SparseGraph<int>, int> readGraph(g, f);

  std::cout << "Test Bellman-Ford:" << std::endl;

  int s = 0;
  BellmanFordSP<SparseGraph<int>, int> bellmanFord(g, s);
  ASSERT_FALSE(bellmanFord.negativeCycle());
  for (int i = 0; i < V; i++) {
    if (i == s) {
      continue;
    }

    if (bellmanFord.hasPathTo(i)) {
      std::cout << "Shortest Path to " << i << " : "
                << bellmanFord.shortestPathTo(i) << std::endl;
      bellmanFord.showPath(i);
    } else {
      std::cout << "No Path to " << i << std::endl;
    }

    std::cout << "----------" << std::endl;
  }
}

TEST_F(BellmanFordSPTest, NegativeCycle) {
  std::string f = (dataDir / "testG9_negative_circle.txt").string();
  int V = 5;
  SparseGraph<int> g = SparseGraph<int>(V, true);
  ReadGraph<SparseGraph<int>, int> readGraph(g, f);

  std::cout << "Test Bellman-Ford:" << std::endl;

  int s = 0;
  BellmanFordSP<SparseGraph<int>, int> bellmanFord(g, s);
  ASSERT_TRUE(bellmanFord.negativeCycle());
}
