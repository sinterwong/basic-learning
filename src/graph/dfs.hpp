#ifndef BL_GRAPH_DEEP_FIRST_SEARCH_HPP
#define BL_GRAPH_DEEP_FIRST_SEARCH_HPP
#include "graph.hpp"
namespace bl::graph {
class GraphDFS {
public:
  GraphDFS(const Graph &graph) : m_graph(graph), m_visited(graph.V(), false) {}

  std::vector<int> operator()(bool isPreOrder = true) {
    m_visited.assign(m_visited.size(), false);

    std::vector<int> ret;
    for (int v = 0; v < m_graph.V(); ++v) {
      if (!m_visited[v]) {
        dfs(v, ret, isPreOrder);
      }
    }
    return ret;
  }

  std::vector<int> operator()(int v, bool isPreOrder = true) {
    m_visited.assign(m_visited.size(), false);

    std::vector<int> ret;
    dfs(v, ret, isPreOrder);
    return ret;
  }

private:
  void dfs(int v, std::vector<int> &ret, bool isPreOrder) {
    if (v > m_graph.V() - 1) {
      throw std::runtime_error("start index is out of bound");
    }
    m_visited[v] = true;
    if (isPreOrder) {
      ret.push_back(v);
    }
    for (const auto &w : m_graph.adj(v)) {
      if (!m_visited[w]) {
        dfs(w, ret, isPreOrder);
      }
    }
    if (!isPreOrder) {
      ret.push_back(v);
    }
  }

private:
  Graph m_graph;
  std::vector<bool> m_visited;
};
} // namespace bl::graph
#endif