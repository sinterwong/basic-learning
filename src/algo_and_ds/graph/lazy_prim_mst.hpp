/**
 * @file lazy_prim_mst.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-02
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __AADS_LAZY_PRIM_MST_HPP_
#define __AADS_LAZY_PRIM_MST_HPP_

#include "min_heap.hpp"
#include "weighted_edge.hpp"
#include <memory>
#include <numeric>
#include <vector>

using algo_and_ds::tree::MinHeap;

namespace algo_and_ds::graph {

template <typename Graph, typename Weight> class LazyPrimMST {

private:
  // 需要获取最小生成树的图
  Graph &G;

  // 使用最小堆获取每次可能的最小生成树的边
  MinHeap<Edge<Weight>> pq;

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

    // 切分并入堆
    for (auto e = iter.begin(); !iter.end(); e = iter.next()) {
      // 如果v顶点的另一端顶点没有被标记过，就加入它们的边
      if (!marked.at(e->other(v))) {
        pq.insert(*e);
      }
    }
  }

public:
  LazyPrimMST(Graph &graph) : G(graph) {
    marked = std::vector<bool>(G.V(), false);
    mst.clear();

    // Lazy Prim
    visit(0); // init

    /* 时间复杂度计算时，要具体问题具体分析，这里的visit看似是和主循环嵌套，但是其实和循环顺序执行的而不是嵌套执行（即相加而不是相乘），
    最终所有的visit调用的时间复杂度总和是O(ElogE)，主循环的时间复杂度是O(ElogE)，因此整个算法的时间复杂度是O(ElogE)
     */

    while (!pq.empty()) {
      auto e = pq.extractMin();

      // 如果两个顶点都已经被标记过了，意味着它们的边已经不是横切边
      if (marked[e.w()] == marked[e.v()]) {
        continue;
      }
      // 加入横切边(crossing edge)
      mst.push_back(e);
      if (!marked[e.v()]) {
        visit(e.v());
      } else {
        visit(e.w());
      }
    }

    mstWeight = mst.at(0).wt();
    for (int i = 1; i < mst.size(); ++i) {
      mstWeight += mst.at(i).wt();
    }
  }

  ~LazyPrimMST() {}

  std::vector<Edge<Weight>> mstEdges() { return mst; }

  Weight weightValue() { return mstWeight; }
};

} // namespace algo_and_ds::graph

#endif