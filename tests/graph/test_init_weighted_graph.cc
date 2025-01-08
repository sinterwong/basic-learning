/**
 * @file test_init_weighted_graph.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-02
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "weighted_dense_graph.hpp"
#include "weighted_read_graph.hpp"
#include "weighted_sparse_graph.hpp"
#include <filesystem>

#include <iomanip>

#include <gtest/gtest.h>

using namespace algo_and_ds::graph;
namespace fs = std::filesystem;

class InitWeightedGraphTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  fs::path dataDir = "data/graph";
};

TEST_F(InitWeightedGraphTest, WeightedSparseGraph) {
  std::string f = (dataDir / "testG3.txt").string();
  int V = 8;
  SparseGraph<double> g = SparseGraph<double>(V, false);
  ReadGraph<SparseGraph<double>, double> readGraph(g, f);
  g.show();
}

TEST_F(InitWeightedGraphTest, WeightedDenseGraph) {
  std::string f = (dataDir / "testG3.txt").string();
  int V = 8;
  DenseGraph<double> g = DenseGraph<double>(V, false);
  ReadGraph<DenseGraph<double>, double> readGraph(g, f);
  g.show();
}
