/**
 * @file sparse_graph.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-01
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __AADS_SPARSE_GRAPH_HPP_
#define __AADS_SPARSE_GRAPH_HPP_

#include <cassert>
#include <vector>

namespace algo_and_ds::graph {
class SparseGraph {
private:
  int n, m; // 图的节点数和边数

  bool isDirected; // 是否是有向图

  std::vector<std::vector<int>> graph;

public:
  SparseGraph(int n, bool isDirected) : n(n), isDirected(isDirected) {
    // 初始化邻接矩阵
    for (int i = 0; i < n; i++) {
      graph.push_back(std::vector<int>());
    }
  }

  int V() { return n; }
  int E() { return m; }

  void addEdge(int v, int w) {
    // 此处没有避免掉平行边，因为判断是否存在平行边的时间复杂度是O(n)
    // if (hasEdge(v, w)) {
    //   return;
    // }

    assert(v >= 0 && v < n);
    assert(w >= 0 && w < n);

    graph[v].push_back(w);
    // v != w 解决环边问题（简单图不考虑环边和平行边）
    if (v != w && !isDirected) {
      graph[w].push_back(v);
    }
    m++;
  }

  bool hasEdge(int v, int w) {
    assert(v >= 0 && v < n);
    assert(w >= 0 && w < n);
    for (int i = 0; i < graph[v].size(); i++) {
      if (graph[v][i] == w) {
        return true;
      }
    }
    return false;
  }
};
} // namespace algo_and_ds::graph

#endif