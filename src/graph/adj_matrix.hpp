#ifndef BL_ADJ_MATRIX_HPP
#define BL_ADJ_MATRIX_HPP

#include <ostream>
#include <string>
#include <vector>

namespace bl::graph {
class AdjMatrix {
public:
  AdjMatrix(const std::string &dataPath);

  friend std::ostream &operator<<(std::ostream &os, const AdjMatrix &adj) {
    os << "V: " << adj.m_V << ", E: " << adj.m_E << std::endl;
    for (int i = 0; i < adj.m_V; ++i) {
      os << i << ": ";
      for (int j = 0; j < adj.m_V; ++j) {
        os << adj.m_adj[i][j] << " ";
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

private:
  void validateVertex(int v) const;

private:
  int m_V;
  int m_E;
  std::vector<std::vector<int>> m_adj;
};
} // namespace bl::graph

#endif