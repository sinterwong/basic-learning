/**
 * @file dense_graph.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-01
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __AADS_WEIGHT_DENSE_GRAPH_HPP_
#define __AADS_WEIGHT_DENSE_GRAPH_HPP_

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include "weighted_edge.hpp"

namespace algo_and_ds::graph {
template <typename Weight> class DenseGraph {

  using weight_edge = Edge<Weight>;
  using weight_edge_ptr = std::shared_ptr<weight_edge>;

  int n, m; // 图的节点数和边数

  bool isDirected;
  std::vector<std::vector<weight_edge_ptr>> graph;

public:
  DenseGraph(int n, bool isDirected) : n(n), isDirected(isDirected) {
    // 初始化邻接矩阵（N x N），边中
    graph = std::vector<std::vector<weight_edge_ptr>>(
        n, std::vector<weight_edge_ptr>(n, nullptr));
  }

  ~DenseGraph() {}

  int V() { return n; }
  int E() { return m; }

  void addEdge(int v, int w, Weight wt) {
    // 因为实现的是简单图，所以需要避免掉平行边
    if (hasEdge(v, w)) {
      // 删除之前的边
      graph[v][w] = nullptr;
      if (!isDirected) {
        graph[w][v] = nullptr;
      }
      m--;
    }

    graph[v][w] = std::make_shared<weight_edge>(v, w, wt);
    if (!isDirected) {
      graph[w][v] = std::make_shared<weight_edge>(v, w, wt);
    }
    // 有向图的话如果两个节点之间互相连接则认为是两个边
    m++;
  }

  bool hasEdge(int v, int w) {
    assert(v >= 0 && v < n);
    assert(w >= 0 && w < n);
    return graph[v][w] != nullptr;
  }

  void show() {
    std::cout << "adjacent matrix: " << std::endl;
    for (int i = 0; i < V(); i++) {
      for (int j = 0; j < V(); j++) {
        if (graph[i][j]) {
          std::cout << graph[i][j]->wt() << ", ";
        } else {
          std::cout << "null"
                    << ", ";
        }
      }
      std::cout << std::endl;
    }
  }

public:
  class adjIterator {
    DenseGraph &G;
    int v;
    int index;

  public:
    adjIterator(DenseGraph &g, int _v) : G(g), v(_v), index(-1) {}

    weight_edge_ptr begin() {
      index = -1;
      return next();
    }

    weight_edge_ptr next() {
      for (index += 1; index < G.V(); index++) {
        if (G.graph[v][index]) {
          return G.graph[v][index];
        }
      }
      return nullptr;
    }

    bool end() { return index >= G.V(); }
  };
};
} // namespace algo_and_ds::graph

#endif