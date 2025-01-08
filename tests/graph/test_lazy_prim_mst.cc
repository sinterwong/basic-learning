/**
 * @file test_lazy_prim_mst.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-02
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "lazy_prim_mst.hpp"
#include "weighted_read_graph.hpp"
#include "weighted_sparse_graph.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <string>

using namespace algo_and_ds::graph;
namespace fs = std::filesystem;

class LazyPrimMSTTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  fs::path dataDir = "data/graph";
};

TEST_F(LazyPrimMSTTest, Normal) {
  std::string f = (dataDir / "testG3.txt").string();
  int V = 8;
  SparseGraph<double> g = SparseGraph<double>(V, false);
  ReadGraph<SparseGraph<double>, double> readGraph(g, f);
  g.show();
  std::cout << std::endl;

  std::cout << "Test Lazy Prim MST: " << std::endl;
  LazyPrimMST<SparseGraph<double>, double> lazyPrimMST(g);
  std::vector<Edge<double>> mst = lazyPrimMST.mstEdges();
  for (int i = 0; i < mst.size(); i++) {
    std::cout << mst[i] << std::endl;
  }
  std::cout << "The MST weight is: " << lazyPrimMST.weightValue() << std::endl;
  std::cout << std::endl;
}
