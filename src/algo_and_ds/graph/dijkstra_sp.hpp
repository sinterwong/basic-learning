/**
 * @file dijkstra_sp.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-05
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __AADS_KRUSKAL_SP_HPP_
#define __AADS_KRUSKAL_SP_HPP_

#include "index_min_heap.hpp"
#include "weighted_edge.hpp"
#include <iostream>
#include <memory>
#include <stack>
#include <vector>
namespace algo_and_ds::graph {

using tree::IndexMinHeap;

template <typename Graph, typename Weight> class DijkstraSP {

  using edge_ptr = std::shared_ptr<Edge<Weight>>;

  Graph &G;
  int s;

  std::vector<Weight> distTo; // 原点抵达每个节点的距离
  std::vector<bool> marked;   // 节点是否已经访问过
  std::vector<edge_ptr> from;

public:
  DijkstraSP(Graph &graph, int _s) : G(graph), s(_s) {
    distTo = std::vector<Weight>(G.V(), Weight());
    marked = std::vector<bool>(G.V(), false);
    from = std::vector<edge_ptr>(G.V(), nullptr);

    IndexMinHeap<Weight> ipq(G.V());

    // Dijkstra
    distTo[s] = Weight();
    from[s] = std::make_shared<Edge<Weight>>(s, s, Weight());
    marked[s] = true;
    ipq.insert(s, distTo[s]);

    while (!ipq.isEmpty()) {
      int v = ipq.extractMinIndex();

      // distTo[v] 就是s到v的最短距离
      marked[v] = true;

      // Relaxation
      typename Graph::adjIterator iter(G, v);
      for (auto e = iter.begin(); !iter.end(); e = iter.next()) {
        int w = e->other(v);
        if (!marked[w]) {
          if (from[w] == nullptr || distTo[v] + e->wt() < distTo[w]) {
            distTo[w] = distTo[v] + e->wt();
            from[w] = e;
            if (ipq.contain(w)) {
              ipq.change(w, distTo[w]);
            } else {
              ipq.insert(w, distTo[w]);
            }
          }
        }
      }
    }
  }

  Weight shortestPathTo(int w) { return distTo.at(w); }

  bool hasPathTo(int w) { return marked.at(w); }

  void shortestPath(int w, std::vector<Edge<Weight>> &vec) {
    std::stack<edge_ptr> st;
    auto e = from.at(w);

    while (e->v() != e->w()) {
      st.push(e);
      e = from[e->v()];
    }

    while (!st.empty()) {
      e = st.top();
      vec.push_back(*e);
      st.pop();
    }
  }

  void showPath(int w) {
    assert(w >= 0 && w < G.V());

    std::vector<Edge<Weight>> vec;
    shortestPath(w, vec);
    for (int i = 0; i < vec.size(); ++i) {
      std::cout << vec[i].v() << " -> ";
      if (i == vec.size() - 1) {
        std::cout << vec[i].w() << std::endl;
      }
    }
  }
};
} // namespace algo_and_ds::graph

#endif