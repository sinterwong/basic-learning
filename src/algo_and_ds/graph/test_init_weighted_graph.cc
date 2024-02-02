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

#include <iomanip>

using namespace algo_and_ds::graph;

int main() {
  std::string filename = "/home/wangxt/workspace/projects/basic-learning/src/"
                         "algo_and_ds/graph/testG3.txt";
  int V = 8;

  // 精确到第二位
  std::cout << std::fixed << std::setprecision(2);

  // Test Weighted Dense Graph
  DenseGraph<double> g1 = DenseGraph<double>(V, false);
  ReadGraph<DenseGraph<double>, double> readGraph1(g1, filename);
  g1.show();
  std::cout << std::endl;

  SparseGraph<double> g2 = SparseGraph<double>(V, false);
  ReadGraph<SparseGraph<double>, double> readGraph2(g2, filename);
  g2.show();

  return 0;
}