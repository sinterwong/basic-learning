/**
 * @file connected_component.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-01
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __AADS_CONNECTED_COMPONENT_HPP_
#define __AADS_CONNECTED_COMPONENT_HPP_

#include <cassert>
#include <vector>

namespace algo_and_ds::graph {
template <typename Graph> class ConnectedComponent {
  Graph &G;
  int ccount = 0; // 几个联通的图

  std::vector<bool> visited; // 每个节点是否已经访问过
  std::vector<int> ids;

public:
  ConnectedComponent(Graph &graph) : G(graph) {
    // 初始化所有的顶点为未访问过的状态
    visited = std::vector<bool>(G.V(), false);
    ids = std::vector<int>(G.V(), -1);

    for (int i = 0; i < G.V(); i++) {
      if (!visited[i]) {
        deepFirstSearch(i);
        // 能存在没有遍历到的节点意味着肯定存在一个新的联通，所以此处更新联通分量
        ccount++;
      }
    }
  }

  int count() { return ccount; }

  bool isConnected(int v, int w) { return ids.at(v) == ids.at(w); }

private:
  void deepFirstSearch(int v) {
    visited.at(v) = true;
    // 这里可以直接使用ccount作为是否是一个组的id
    ids.at(v) = ccount;
    typename Graph::adjIterator adj(G, v);
    for (int i = adj.begin(); !adj.end(); i = adj.next()) {
      if (!visited.at(i)) {
        deepFirstSearch(i);
      }
    }
  }
};

} // namespace algo_and_ds::graph
#endif