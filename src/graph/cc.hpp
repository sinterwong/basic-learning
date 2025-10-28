#ifndef BL_GRAPH_CONNECTED_COMPONENT_HPP
#define BL_GRAPH_CONNECTED_COMPONENT_HPP
#include "graph.hpp"
#include <memory>
namespace bl::graph {
class ConnectedComponent {
public:
  ConnectedComponent(const Graph &graph)
      : m_visited(graph.V(), -1), m_ccCount(0) {
    m_graph = std::make_unique<Graph>(graph);
    for (int v = 0; v < m_graph->V(); ++v) {
      if (m_visited[v] < 0) {
        dfs(v, m_ccCount++);
      }
    }
  }

  int32_t count() const noexcept { return m_ccCount; }

  std::vector<std::vector<int32_t>> components() const {
    std::vector<std::vector<int32_t>> rets(m_ccCount);
    for (int32_t i = 0; i < m_visited.size(); ++i) {
      rets[m_visited.at(i)].push_back(i);
    }
    return rets;
  }

  bool isConnected(int32_t v, int32_t w) const {
    m_graph->validateVertex(v);
    m_graph->validateVertex(w);
    return m_visited[v] == m_visited[w];
  }

private:
  void dfs(int32_t v, int32_t ccid) {
    if (v > m_graph->V() - 1) {
      throw std::runtime_error("start index is out of bound");
    }
    m_visited[v] = ccid;
    for (const auto &w : m_graph->adj(v)) {
      if (m_visited[w] < 0) {
        dfs(w, ccid);
      }
    }
  }

private:
  std::unique_ptr<Graph> m_graph;
  int32_t m_ccCount;
  std::vector<int32_t> m_visited;
};
} // namespace bl::graph
#endif