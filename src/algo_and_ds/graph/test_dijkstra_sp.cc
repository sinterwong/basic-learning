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
#include <string>

using namespace algo_and_ds::graph;

int main() {
  std::string filename = "/home/wangxt/workspace/projects/basic-learning/src/"
                         "algo_and_ds/graph/testG7.txt";
  int V = 5;
  SparseGraph<int> g = SparseGraph<int>(V, true);
  ReadGraph<SparseGraph<int>, int> readGraph(g, filename);
  g.show();
  std::cout << std::endl;

  std::cout << "Test Dijkstra SP: " << std::endl;
  DijkstraSP<SparseGraph<int>, int> dijkstraSP(g, 0);
  for (int i = 0; i < V; i++) {
    if (dijkstraSP.hasPathTo(i)) {
      std::cout << "Shortest Path to " << i << " : "
                << dijkstraSP.shortestPathTo(i) << std::endl;
      dijkstraSP.showPath(i);
    } else
      std::cout << "No Path to " << i << std::endl;

    std::cout << "----------" << std::endl;
  }
  return 0;
}