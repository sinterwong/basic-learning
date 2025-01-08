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
#include "connected_component.hpp"
#include "dense_graph.hpp"
#include "read_graph.hpp"
#include "sparse_graph.hpp"
#include <filesystem>
#include <gtest/gtest.h>

using namespace algo_and_ds::graph;
namespace fs = std::filesystem;

class ConnectedComponentTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  fs::path dataDir = "data/graph";
};

TEST_F(ConnectedComponentTest, Normal) {
  std::string f = (dataDir / "testG1.txt").string();
  SparseGraph g(13, false);
  ReadGraph<SparseGraph> readGraph(g, f);
  ConnectedComponent<SparseGraph> cc(g);
  ASSERT_EQ(cc.count(), 3);
}

TEST_F(ConnectedComponentTest, Normal2) {
  std::string f = (dataDir / "testG2.txt").string();
  DenseGraph g(7, false);
  ReadGraph<DenseGraph> readGraph(g, f);
  ConnectedComponent<DenseGraph> cc(g);
  ASSERT_EQ(cc.count(), 1);
};
