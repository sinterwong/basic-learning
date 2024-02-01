/**
 * @file test_read_graph.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-01
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "dense_graph.hpp"
#include "read_graph.hpp"
#include "sparse_graph.hpp"

using namespace algo_and_ds::graph;

int main() {
  std::string filename = "/home/wangxt/workspace/projects/basic-learning/src/"
                         "algo_and_ds/graph/testG1.txt";
  SparseGraph g1(13, false);
  ReadGraph<SparseGraph> readGraph01(g1, filename);

  g1.show();

  std::cout << std::endl;

  DenseGraph g2(13, false);
  ReadGraph<DenseGraph> readGraph02(g2, filename);

  g2.show();
}