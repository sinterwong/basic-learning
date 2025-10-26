/**
 * @file test_dijkstra_sp.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-05
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "dijkstra_sp.hpp"
#include "weighted_read_graph.hpp"
#include "weighted_sparse_graph.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <string>

using namespace algo_and_ds::graph;
namespace fs = std::filesystem;

class DijkstraSPTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  fs::path dataDir = "assets/data/graph";
};

TEST_F(DijkstraSPTest, Normal) {
  std::string f = (dataDir / "testG7.txt").string();

  int V = 5;
  SparseGraph<int> g = SparseGraph<int>(V, true);
  ReadGraph<SparseGraph<int>, int> readGraph(g, f);

  std::cout << "Test Dijkstra:" << std::endl << std::endl;

  int s = 0;
  DijkstraSP<SparseGraph<int>, int> dijkstra(g, s);
  for (int i = 0; i < V; i++) {
    if (i == s) {
      continue;
    }

    if (dijkstra.hasPathTo(i)) {
      std::cout << "Shortest Path to " << i << " : "
                << dijkstra.shortestPathTo(i) << std::endl;
      dijkstra.showPath(i);
    } else {
      std::cout << "No Path to " << i << std::endl;
    }

    std::cout << "----------" << std::endl;
  }
}
