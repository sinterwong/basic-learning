/**
 * @file test_connected_component.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-01
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "bfs_path.hpp"
#include "dfs_path.hpp"
#include "read_graph.hpp"
#include "sparse_graph.hpp"
#include <filesystem>
#include <gtest/gtest.h>

using namespace algo_and_ds::graph;
namespace fs = std::filesystem;

class PathTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  fs::path dataDir = "assets/data/graph";
};

TEST_F(PathTest, Normal) {
  std::string f = (dataDir / "testG2.txt").string();
  SparseGraph g(7, false);
  ReadGraph<SparseGraph> readGraph(g, f);
  g.show();
  std::cout << std::endl;

  Path<SparseGraph> path(g, 0);
  std::cout << "DFS: ";
  path.showPath(6);

  ShortestPath<SparseGraph> shortestPath(g, 0);
  std::cout << "BFS: ";
  shortestPath.showPath(6);
}
