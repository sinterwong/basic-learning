/**
 * @file test_read_graph.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-01
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "dense_graph.hpp"
#include "read_graph.hpp"
#include "sparse_graph.hpp"

#include <filesystem>
#include <gtest/gtest.h>

using namespace algo_and_ds::graph;
namespace fs = std::filesystem;

class ReadGraphTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  fs::path dataDir = "data/graph";
};

TEST_F(ReadGraphTest, SparseGraph) {
  std::string f = (dataDir / "testG1.txt").string();
  SparseGraph g(13, false);
  ReadGraph<SparseGraph> readGraph(g, f);
  g.show();
}

TEST_F(ReadGraphTest, DenseGraph) {
  std::string f = (dataDir / "testG1.txt").string();
  DenseGraph g(13, false);
  ReadGraph<DenseGraph> readGraph(g, f);
  g.show();
}
