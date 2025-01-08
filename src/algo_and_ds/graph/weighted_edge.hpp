/**
 * @file weight_edge.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-02
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __AADS_WEIGHT_EDGE_HPP_
#define __AADS_WEIGHT_EDGE_HPP_
#include <cassert>
#include <ostream>
namespace algo_and_ds::graph {

template <typename Weight> class Edge {
  int _a, _b; // 两个定点
  Weight _weight;

public:
  Edge(int a, int b, Weight weight) : _a(a), _b(b), _weight(weight) {}

  Edge(){};

  int v() { return _a; }

  int w() { return _b; }

  Weight wt() { return _weight; }

  int other(int x) {
    assert(x == _a || x == _b);
    // 根据其中一个定点获取另外的一个定点
    return x == _a ? _b : _a;
  }
  // 加上friend关键字，该函数实际上就不是类的成员函数了
  friend std::ostream &operator<<(std::ostream &os, Edge const &e) {
    os << e._a << "-" << e._b << ": " << e._weight;
    return os;
  }

  bool operator<(Edge<Weight> &e) { return _weight < e.wt(); }

  bool operator>(Edge<Weight> &e) { return _weight > e.wt(); }

  bool operator<=(Edge<Weight> &e) { return _weight <= e.wt(); }

  bool operator>=(Edge<Weight> &e) { return _weight >= e.wt(); }

  bool operator==(Edge<Weight> &e) { return _weight == e.wt(); }
};
} // namespace algo_and_ds::graph

#endif