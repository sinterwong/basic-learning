/**
 * @file test_graph.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-01
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "dense_graph.hpp"
#include "sparse_graph.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace algo_and_ds::graph;

int main() {
  int N = 20;
  int M = 100;

  srand(time(nullptr));

  // Sparse Graph
  SparseGraph g1(N, false);

  for (int i = 0; i < M; i++) {
    int a = rand() % N;
    int b = rand() % N;

    g1.addEdge(a, b);
  }

  for (int v = 0; v < N; v++) {
    std::cout << v << ": ";
    SparseGraph::adjIterator adj(g1, v);
    for (int w = adj.begin(); !adj.end(); w = adj.next()) {
      std::cout << w << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // Dense Graph
  DenseGraph g2(N, false);

  for (int i = 0; i < M; i++) {
    int a = rand() % N;
    int b = rand() % N;

    g2.addEdge(a, b);
  }

  for (int v = 0; v < N; v++) {
    std::cout << v << ": ";
    DenseGraph::adjIterator adj(g2, v);
    for (int w = adj.begin(); !adj.end(); w = adj.next()) {
      std::cout << w << ", ";
    }
    std::cout << std::endl;
  }
}