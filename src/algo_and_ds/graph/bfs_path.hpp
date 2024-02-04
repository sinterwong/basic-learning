/**
 * @file bfs_path.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-01
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __AADS_BFS_PATH_HPP_
#define __AADS_BFS_PATH_HPP_

#include <cassert>
#include <iostream>
#include <memory>
#include <queue>
#include <stack>
#include <vector>

namespace algo_and_ds::graph {
template <typename Graph> class ShortestPath {
  Graph &G;
  int source; // 起点节点
  std::unique_ptr<std::vector<bool>> visited;
  std::unique_ptr<std::vector<int>> from; // 每个节点的上家
  std::unique_ptr<std::vector<int>> ord;  // 距离源节点的距离

public:
  ShortestPath(Graph &graph, int s) : G(graph), source(s) {
    assert(s >= 0 && s < G.V());
    visited = std::make_unique<std::vector<bool>>(G.V(), false);
    from = std::make_unique<std::vector<int>>(G.V(), -1);
    ord = std::make_unique<std::vector<int>>(G.V(), -1);

    std::queue<int> q;

    // 无向图最短路径算法
    q.push(s);
    visited->at(s) = true;
    ord->at(s) = 0; // 源节点距离自己为0
    while (!q.empty()) {
      int v = q.front();
      q.pop();

      typename Graph::adjIterator iter(G, s);
      for (int i = iter.begin(); !iter.end(); i = iter.next()) {
        if (!visited->at(i)) {
          q.push(i);
          visited->at(i) = true;
          from->at(i) = v;
          ord->at(i) = ord->at(v) + 1;
        }
      }
    }
  }

public:
  bool hasPath(int w) {
    // 只要访问到了w，说明s->w的路是通的
    return visited->at(w);
  }

  void path(int w, std::vector<int> &vec) {
    int p = w;

    // 倒着存储到栈
    std::stack<int> s;
    while (p != -1) {
      s.push(p);
      // 更新p为p的上一个节点
      p = from->at(p);
    }
    vec.clear();
    while (!s.empty()) {
      vec.push_back(s.top());
      s.pop();
    }
  }

  void showPath(int w) {
    std::vector<int> vec;
    path(w, vec);
    for (int i = 0; i < vec.size(); i++) {
      std::cout << vec[i];
      if (i == vec.size() - 1) {
        std::cout << std::endl;
      } else {
        std::cout << "->";
      }
    }
  }

  int length(int w) { return ord->at(w); }
};

} // namespace algo_and_ds::graph

#endif