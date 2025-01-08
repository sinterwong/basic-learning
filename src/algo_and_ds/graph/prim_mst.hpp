/**
 * @file prim_mst.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-04
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __AADS_PRIM_MST_HPP_
#define __AADS_PRIM_MST_HPP_

#include "index_min_heap.hpp"
#include "weighted_edge.hpp"
#include <memory>
#include <numeric>
#include <vector>

using algo_and_ds::tree::IndexMinHeap;

namespace algo_and_ds::graph {

template <typename Graph, typename Weight> class PrimMST {

  using edge_ptr = std::shared_ptr<Edge<Weight>>;

  // 需要获取最小生成树的图
  Graph &G;

  // 使用最小堆获取每次可能的最小生成树的边
  IndexMinHeap<Weight> ipq;

  // 存储和每个节点相邻的最短横切边
  std::vector<edge_ptr> edgeTo;

  // 节点是否已经访问过
  std::vector<bool> marked;

  // 最小生成树的边
  std::vector<Edge<Weight>> mst;

  // 最小生成树的权值
  Weight mstWeight;

private:
  void visit(int v) {
    assert(!marked.at(v));
    marked[v] = true;

    typename Graph::adjIterator iter(G, v);

    // 迭代该节点对应的所有边（以此为点横切后的边）
    for (auto e = iter.begin(); !iter.end(); e = iter.next()) {
      int w = e->other(v); // 另一端顶点
      // 另一端还没有标记的情况下需要考虑入队
      if (!marked[w]) {
        if (!edgeTo[w]) {
          // 此时还不存在另一个它俩的边
          ipq.insert(w, e->wt());
          edgeTo[w] = e;
        } else if (e->wt() < edgeTo[w]->wt()) {
          edgeTo[w] = e;
          ipq.change(w, e->wt());
        }
      }
    }
  }

public:
  PrimMST(Graph &graph) : G(graph), ipq(IndexMinHeap<Weight>(G.V())) {
    marked = std::vector<bool>(G.V(), false);
    edgeTo = std::vector<edge_ptr>(G.V(), nullptr);

    mst.clear();

    // Prim MST
    visit(0);

    while (!ipq.isEmpty()) {
      int v = ipq.extractMinIndex();
      assert(edgeTo[v]);

      // 获取最小横切边
      mst.push_back(*edgeTo[v]);

      // 将节点加入已完成获取最小横切边的阵营，继续切分
      visit(v);
    }

    mstWeight = mst.at(0).wt();
    for (int i = 1; i < mst.size(); ++i) {
      mstWeight += mst.at(i).wt();
    }
  }

  ~PrimMST() {}

  std::vector<Edge<Weight>> mstEdges() { return mst; }

  Weight weightValue() { return mstWeight; }
};

} // namespace algo_and_ds::graph

#endif