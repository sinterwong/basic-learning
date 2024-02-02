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

  std::cout << "Test Lazy Prim MST: " << std::endl;
  LazyPrimMST<SparseGraph<double>, double> lazyPrimMST(g);
  std::vector<Edge<double>> mst = lazyPrimMST.mstEdges();
  for (int i = 0; i < mst.size(); i++) {
    std::cout << mst[i] << std::endl;
  }
  std::cout << "The MST weight is: " << lazyPrimMST.weightValue() << std::endl;
  std::cout << std::endl;
  return 0;
}