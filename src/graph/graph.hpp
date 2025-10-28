#ifndef BL_GRAPH_HPP
#define BL_GRAPH_HPP

#include <ostream>
#include <set>
#include <string>
#include <vector>

namespace bl::graph {
class Graph {
public:
  Graph(const std::string &dataPath);

  friend std::ostream &operator<<(std::ostream &os, const Graph &adj) {
    os << "V: " << adj.m_V << ", E: " << adj.m_E << std::endl;
    for (int i = 0; i < adj.m_adj.size(); ++i) {
      os << i << ": ";
      for (const auto &w : adj.m_adj.at(i)) {
        os << w << " ";
      }
      os << std::endl;
    }
    return os;
  }

public:
  int V() const noexcept;
  int E() const noexcept;
  bool hasEdge(int v, int w) const;
  std::vector<int> adj(int v) const;
  int degree(int v) const;
  void validateVertex(int v) const;

private:
  int m_V;
  int m_E;
  std::vector<std::set<int>> m_adj;
};
} // namespace bl::graph

#endif