/**
 * @file weight_read_graph.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-02
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __AADS_WEIGHT_READ_GRAPH_HPP_
#define __AADS_WEIGHT_READ_GRAPH_HPP_

#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
namespace algo_and_ds::graph {
template <typename Graph, typename Weight> class ReadGraph {
public:
  ReadGraph(Graph &graph, std::string const &filename) {
    std::ifstream file(filename);
    std::string line;
    int V, E;

    assert(file.is_open());
    assert(std::getline(file, line));

    std::stringstream ss(line);
    ss >> V >> E;

    assert(V == graph.V());

    for (int i = 0; i < E; ++i) {
      assert(std::getline(file, line));
      std::stringstream ss(line);

      int a, b;
      Weight w;
      ss >> a >> b >> w;
      assert(a >= 0 && a < V);
      assert(b >= 0 && b < V);

      graph.addEdge(a, b, w);
    }
  }
};

} // namespace algo_and_ds::graph
#endif