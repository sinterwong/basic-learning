#include "graph.hpp"
#include <fstream>
#include <logger.hpp>
#include <sstream>

namespace bl::graph {
Graph::Graph(const std::string &dataPath) {
  std::ifstream ff(dataPath);
  if (!ff.is_open()) {
    LOG_ERRORS << "Unable to open file: " << dataPath;
    throw std::runtime_error("Unable to open file: " + dataPath);
  }
  std::string line;
  if (std::getline(ff, line)) {
    std::stringstream ss(line);
    ss >> m_V >> m_E;
  } else {
    LOG_ERRORS << "Unable to read V and E from file: " << dataPath;
    throw std::runtime_error("Unable to read V and E from file: " + dataPath);
  }

  if (m_V < 0) {
    LOG_ERRORS << "V cannot be negative: " << m_V;
    throw std::runtime_error("V cannot be negative: " + std::to_string(m_V));
  }
  if (m_E < 0) {
    LOG_ERRORS << "E cannot be negative: " << m_E;
    throw std::runtime_error("E cannot be negative: " + std::to_string(m_E));
  }

  m_adj.resize(m_V);
  try {
    for (int i = 0; i < m_E; ++i) {
      std::getline(ff, line);
      std::stringstream ss(line);
      int a, b;
      ss >> a >> b;
      validateVertex(a);
      validateVertex(b);

      if (a == b) {
        LOG_ERRORS << "Self-loops are not allowed: a = b = " << a;
        throw std::runtime_error("Self-loops are not allowed.");
      }

      if (m_adj.at(a).count(b)) {
        LOG_ERRORS << "Parallel edges are not allowed: a=" << a << ", b=" << b;
        throw std::runtime_error("Parallel edges are not allowed.");
      }

      m_adj.at(a).insert(b);
      m_adj.at(b).insert(a);
    }
  } catch (const std::exception &e) {
    LOG_ERRORS << "Error reading edge data: " << e.what();
    throw std::runtime_error("Error reading edge data: " +
                             std::string(e.what()));
  }
  ff.close();
}

void Graph::validateVertex(int v) const {
  if (v < 0 || v > m_V) {
    throw std::runtime_error("Vertex " + std::to_string(v) +
                             " is out of bounds.");
  }
}

int Graph::V() const noexcept { return m_V; }

int Graph::E() const noexcept { return m_E; }

bool Graph::hasEdge(int v, int w) const {
  validateVertex(v);
  validateVertex(w);
  return m_adj[v].count(w);
}

std::vector<int> Graph::adj(int v) const {
  validateVertex(v);
  auto retSet = m_adj.at(v);
  std::vector<int> ret(retSet.begin(), retSet.end());
  return ret;
}

int Graph::degree(int v) const { return adj(v).size(); }

} // namespace bl::graph