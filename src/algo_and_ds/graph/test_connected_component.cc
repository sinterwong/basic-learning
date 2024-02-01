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

using namespace algo_and_ds::graph;

int main() {
  std::string filename1 = "/home/wangxt/workspace/projects/basic-learning/src/"
                          "algo_and_ds/graph/testG1.txt";
  SparseGraph g1(13, false);
  ReadGraph<SparseGraph> readGraph01(g1, filename1);
  ConnectedComponent<SparseGraph> component1(g1);
  std::cout << "TestG1.txt, Connected Component Count: " << component1.count()
            << std::endl;

  std::string filename2 = "/home/wangxt/workspace/projects/basic-learning/src/"
                          "algo_and_ds/graph/testG2.txt";
  DenseGraph g2(7, false);
  ReadGraph<DenseGraph> readGraph02(g2, filename2);
  ConnectedComponent<DenseGraph> component2(g2);
  std::cout << "TestG2.txt, Connected Component Count: " << component2.count()
            << std::endl;

  return 0;
}