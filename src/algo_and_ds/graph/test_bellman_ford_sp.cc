/**
 * @file test_bellman_ford_sp.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-05
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "bellman_ford_sp.hpp"
#include "weighted_read_graph.hpp"
#include "weighted_sparse_graph.hpp"

using namespace algo_and_ds::graph;
using namespace std;

int main() {
  std::string f = "/home/wangxt/workspace/projects/basic-learning/src/"
                  "algo_and_ds/graph/testG8.txt";
  // std::string f = "/home/wangxt/workspace/projects/basic-learning/src/"
  //                 "algo_and_ds/graph/testG9_negative_circle.txt";
  int V = 5;
  SparseGraph<int> g = SparseGraph<int>(V, true);
  ReadGraph<SparseGraph<int>, int> readGraph(g, f);

  cout << "Test Bellman-Ford:" << endl << endl;

  int s = 0;
  BellmanFordSP<SparseGraph<int>, int> bellmanFord(g, s);
  if (bellmanFord.negativeCycle())
    cout << "The graph contain negative cycle!" << endl;
  else
    for (int i = 0; i < V; i++) {
      if (i == s)
        continue;

      if (bellmanFord.hasPathTo(i)) {
        cout << "Shortest Path to " << i << " : "
             << bellmanFord.shortestPathTo(i) << endl;
        bellmanFord.showPath(i);
      } else
        cout << "No Path to " << i << endl;

      cout << "----------" << endl;
    }
  return 0;
}