/**
 * @file test_kruskal_mst.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-05
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "kruskal_mst.hpp"
#include "weighted_read_graph.hpp"
#include "weighted_sparse_graph.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <string>

using namespace algo_and_ds::graph;
namespace fs = std::filesystem;

class KruskalMSTTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  fs::path dataDir = "assets/data/graph";
};

TEST_F(KruskalMSTTest, Normal) {
  std::string f = (dataDir / "testG3.txt").string();
  int V = 8;
  SparseGraph<double> g = SparseGraph<double>(V, false);
  ReadGraph<SparseGraph<double>, double> readGraph(g, f);
  g.show();
  std::cout << std::endl;

  std::cout << "Test Kruskal MST: " << std::endl;
  KruskalMST<SparseGraph<double>, double> kruskalMST(g);
  std::vector<Edge<double>> mst = kruskalMST.mstEdges();
  for (int i = 0; i < mst.size(); i++) {
    std::cout << mst[i] << std::endl;
  }
  std::cout << "The MST weight is: " << kruskalMST.weightValue() << std::endl;
  std::cout << std::endl;
}
