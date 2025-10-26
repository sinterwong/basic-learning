/**
 * @file bellman_ford_sp.hpp
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

#include "weighted_edge.hpp"
#include <iostream>
#include <memory>
#include <stack>
#include <vector>
namespace algo_and_ds::graph {

template <typename Graph, typename Weight> class BellmanFordSP {
  using edge_ptr = std::shared_ptr<Edge<Weight>>;

  Graph &G;
  int s;
  std::vector<Weight> distTo;
  std::vector<edge_ptr> from;
  bool hasNegativeCycle;

public:
  BellmanFordSP(Graph &graph, int _s) : G(graph), s(_s) {
    distTo = std::vector<Weight>(G.V(), Weight());
    from = std::vector<edge_ptr>(G.V(), nullptr);

    // Bellman-Ford
    distTo[s] = Weight();
    from[s] = std::make_shared<Edge<Weight>>(s, s, Weight());

    for (int pass = 1; pass < G.V(); pass++) {

      // Relaxation
      for (int i = 0; i < G.V(); i++) {
        typename Graph::adjIterator iter(G, i);
        for (auto e = iter.begin(); !iter.end(); e = iter.next()) {
          if (!from[e->w()] || distTo[e->v()] + e->wt() < distTo[e->w()]) {
            distTo[e->w()] = distTo[e->v()] + e->wt();
            from[e->w()] = e;
          }
        }
      }
    }

    hasNegativeCycle = detectNegativeCycle();
  }

  Weight shortestPathTo(int w) { return distTo.at(w); }

  bool hasPathTo(int w) { return from.at(w) != nullptr; }

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

  bool negativeCycle() { return hasNegativeCycle; }

private:
  bool detectNegativeCycle() {
    for (int i = 0; i < G.V(); i++) {
      typename Graph::adjIterator iter(G, i);
      for (auto e = iter.begin(); !iter.end(); e = iter.next()) {
        if (!from[e->w()] || distTo[e->v()] + e->wt() < distTo[e->w()]) {
          return true;
        }
      }
    }
    return false;
  }
};

} // namespace algo_and_ds::graph

#endif