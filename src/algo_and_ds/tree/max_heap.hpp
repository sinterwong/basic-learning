/**
 * @file max_heap.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * 最大堆，两个特性：
 * 1. 结构为完全二叉树（除了最后一层节点之外，其他层的节点数量都必须是最大值）
 * 2. 任何一个子节点都不大于它的父节点
 * 优势：入队和出队操作时间复杂度都是O(logn)级别
 * @version 0.1
 * @date 2022-10-11
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __AADS_MAX_HEAP_HPP_
#define __AADS_MAX_HEAP_HPP_
#include <utility>
#include <vector>

namespace algo_and_ds {
namespace tree {

template <typename Item> class MaxHeap {
private:
  std::vector<Item> data;

  void shiftUp(size_t k) {
    // 向上和父节点比较，如果大于父节点就交换位置，父节点计算方式为 int(子节点 /
    // 2)
    while (k > 1 && data[k] > data[k / 2]) {
      std::swap(data[k], data[k / 2]);
      k /= 2; // 继续向上找
    }
  }

  void shiftDown(size_t k) {
    // 向下和最大的子节点比较，直到子节点都小于自己为止
    // 左子节点 = 2 * 父节点，右子节点 = 2 * 父节点 + 1
    while (2 * k <= data.size() - 1) {
      int j = 2 * k; // 用左节点初始化
      if (j < data.size() - 1 && data[j] < data[j + 1]) {
        // 此时存在右节点并且右节点大于左节点
        j++;
      }
      if (data[k] >= data[j]) {
        break;
      }
      std::swap(data[k], data[j]);
      k = j;
    }
  }

public:
  MaxHeap() {
    data.emplace_back(Item()); // 为了更加清楚的计算索引，数据从索引1开始存
  };
  // template <typename Container> MaxHeap(Container _data) {
  // }

  void insert(Item d) {
    data.emplace_back(d);
    // 传入新插入元素的坐标，新元素自顶向上的找到合适的位置
    shiftUp(data.size() - 1);
  }

  Item extractMax() {
    Item ret = data[1];
    data[1] = data.back();
    data.pop_back();
    // 自顶向下将元素放置满足定义的位置
    shiftDown(1);
    return ret;
  }
};

} // namespace tree
} // namespace algo_and_ds

#endif