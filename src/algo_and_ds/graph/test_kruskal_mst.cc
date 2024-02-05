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
#include <string>

using namespace algo_and_ds::graph;

int main() {
  std::string filename = "/home/wangxt/workspace/projects/basic-learning/src/"
                         "algo_and_ds/graph/testG3.txt";
  int V = 8;
  SparseGraph<double> g = SparseGraph<double>(V, false);
  ReadGraph<SparseGraph<double>, double> readGraph2(g, filename);
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
  return 0;
}