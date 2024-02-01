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
#include "bfs_path.hpp"
#include "dfs_path.hpp"
#include "read_graph.hpp"
#include "sparse_graph.hpp"

using namespace algo_and_ds::graph;

int main() {

  std::string filename = "/home/wangxt/workspace/projects/basic-learning/src/"
                         "algo_and_ds/graph/testG2.txt";
  SparseGraph g(7, false);
  ReadGraph<SparseGraph> readGraph(g, filename);
  g.show();
  std::cout << std::endl;

  Path<SparseGraph> path(g, 0);
  std::cout << "DFS: ";
  path.showPath(6);

  ShortestPath<SparseGraph> shortestPath(g, 0);
  std::cout << "BFS: ";
  shortestPath.showPath(6);
  return 0;
}