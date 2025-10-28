#ifndef BL_GRAPH_CONNECTED_PATH_HPP
#define BL_GRAPH_CONNECTED_PATH_HPP
#include "graph.hpp"
#include <algorithm>
namespace bl::graph {
class ConnectedPath {
public:
  ConnectedPath(const Graph &graph) : m_graph(graph) {}

  std::vector<int> path(int s, int t) {
    std::vector<bool> visited(m_graph.V(), false);
    std::vector<int> pre(m_graph.V(), -1);
    dfs(s, s, pre, visited);

    if (pre[t] < 0) { // 不连通
      return {};
    }

    std::vector<int32_t> ret;
    int32_t cur = t;
    while (cur != s) {
      ret.push_back(cur);
      cur = pre[cur];
    }

    std::reverse(ret.begin(), ret.end());
    return ret;
  }

private:
  void dfs(int v, int parent, std::vector<int32_t> &pre,
           std::vector<bool> &visited) {
    if (v > m_graph.V() - 1) {
      throw std::runtime_error("start index is out of bound");
    }
    visited[v] = true;
    pre[v] = parent;
    for (const auto &w : m_graph.adj(v)) {
      if (!visited[w]) {
        dfs(w, v, pre, visited);
      }
    }
  }

private:
  Graph m_graph;
};
} // namespace bl::graph
#endif