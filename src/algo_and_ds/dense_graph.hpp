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
#ifndef __AADS_DENSE_GRAPH_HPP_
#define __AADS_DENSE_GRAPH_HPP_

#include <cassert>
#include <vector>

namespace algo_and_ds::graph {
class DenseGraph {
private:
  int n, m; // 图的节点数和边数

  bool isDirected;
  std::vector<std::vector<int>> graph;

public:
  DenseGraph(int n, bool isDirected) : n(n), isDirected(isDirected) {
    // 初始化邻接矩阵
    for (int i = 0; i < n; i++) {
      graph.push_back(std::vector<int>(n, false));
    }
  }

  int V() { return n; }
  int E() { return m; }

  void addEdge(int v, int w) {
    // 因为实现的是简单图，所以需要避免掉平行边
    if (hasEdge(v, w)) {
      return;
    }
    graph[v][w] = true;
    if (!isDirected) {
      graph[w][v] = true;
    }
    // 有向图的话如果两个节点之间互相连接则认为是两个边
    m++;
  }

  bool hasEdge(int v, int w) {
    assert(v >= 0 && v < n);
    assert(w >= 0 && w < n);
    return graph[v][w];
  }

  void removeAllParallelEdges() {}
};
} // namespace algo_and_ds::graph

#endif