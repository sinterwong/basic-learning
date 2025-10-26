/**
 * @file kruskal_mst.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-05
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __AADS_KRUSKAL_MST_HPP_
#define __AADS_KRUSKAL_MST_HPP_

#include "min_heap.hpp"
#include "union_find.hpp"
#include "weighted_edge.hpp"
#include <memory>

namespace algo_and_ds::graph {

using tree::MinHeap;
using tree::UnionFindTreeOpRank;

template <typename Graph, typename Weight> class KruskalMST {

  std::vector<Edge<Weight>> mst;
  Weight mstWeight;

public:
  KruskalMST(Graph &graph) {

    // 使用堆排序
    MinHeap<Edge<Weight>> pq;
    for (int v = 0; v < graph.V(); v++) {
      typename Graph::adjIterator iter(graph, v);
      for (auto e = iter.begin(); !iter.end(); e = iter.next()) {
        if (e->v() < e->w()) {
          pq.insert(*e);
        }
      }
    }

    UnionFindTreeOpRank uf(graph.V());
    while (!pq.empty()) {
      auto e = pq.extractMin();
      if (uf.isConnection(e.v(), e.w())) {
        continue;
      }
      mst.push_back(e);
      uf.unionElements(e.v(), e.w());
    }

    mstWeight = mst.at(0).wt();
    for (int i = 1; i < mst.size(); ++i) {
      mstWeight += mst.at(i).wt();
    }
  }

  ~KruskalMST() {}

  std::vector<Edge<Weight>> mstEdges() { return mst; }

  Weight weightValue() { return mstWeight; }
};

} // namespace algo_and_ds::graph
#endif