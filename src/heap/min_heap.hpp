/**
 * @file min_heap.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-01-26
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __AADS_MIN_HEAP_HPP_
#define __AADS_MIN_HEAP_HPP_

#include <cassert>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <vector>
namespace algo_and_ds::tree {

template <typename Item> class MinHeap {
private:
  std::vector<Item> data;

  /**
   * @brief 将节点通过向上与父节点比较以更新到合适的位置
   *
   * @param k 待更新元素的初始索引
   */
  void shiftUp(size_t k) {
    // 向上与自己的父节点进行比较
    Item temp = data.at(k);
    while (k > 1 && data.at(k / 2) > temp) {
      data.at(k) = data.at(k / 2);
      k /= 2;
    }
    data.at(k) = temp;
  }

  /**
   * @brief 将节点通过向下与父节点比较以更新到合适的位置
   *
   * @param k 待更新元素的初始索引
   */
  void shiftDown(size_t k) {
    // 向下与自己的子节点进行比较
    while (k * 2 < data.size()) { // 只要k还有左节点
      int j = k * 2;              // 初始孩子节点为左节点
      if (j + 1 < data.size() && data.at(j + 1) < data.at(j)) {
        // 意味着存在右子树，且右子树小于左子树
        j++;
      }

      if (data.at(k) <= data.at(j)) {
        break;
      }

      // std::swap(data.at(k), data.at(j));
      auto temp = data.at(k);
      data.at(k) = data.at(j);
      data.at(j) = temp;
      k = j;
    }
  }

public:
  MinHeap() {
    data.push_back(Item()); // 维护的索引从1开始
  }

  void insert(Item d) {
    data.push_back(d);
    shiftUp(data.size() - 1);
  }

  Item extractMin() {
    assert(data.size() > 1);
    Item ret = data.at(1);
    data.at(1) = data.back();
    data.pop_back();

    shiftDown(1);
    return ret;
  }

  void printData() {
    for (int i = 1; i < data.size(); ++i) {
      std::cout << data.at(i) << ", ";
    }
    std::cout << std::endl;
  }

  bool empty() { return data.size() == 1; }

  int size() { return data.size() - 1; }
};

} // namespace algo_and_ds::tree
#endif