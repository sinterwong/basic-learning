/**
 * @file index_min_heap.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-04
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __AADS_INDEX_MIN_HEAP_HPP_
#define __AADS_INDEX_MIN_HEAP_HPP_
#include <cassert>
#include <tuple>
#include <vector>
namespace algo_and_ds::tree {
template <typename Item> class IndexMinHeap {
  // 数据的配套索引
  std::vector<int> indexes;

  // 反向查找表（记录数据的配套索引在堆数组中的位置）
  std::vector<int> reverse;

  // 堆中数据
  std::vector<Item> datas;

  // 当前堆中的数据量
  int count = 0;

public:
  IndexMinHeap(int capacity) {
    datas.resize(capacity + 1);
    indexes.resize(capacity + 1);
    reverse.resize(capacity + 1);
  }

  int size() { return count; };

  bool isEmpty() { return count == 0; }

  void insert(int i, Item d) {
    assert(count + 1 <= indexes.size() - 1);
    assert(i >= 0 && i + 1 <= indexes.size() - 1);
    // 该数据结构维护的索引是从1开始的
    i++;
    datas[i] = d;
    indexes[++count] = i;
    reverse[i] = count; // i索引 在索引数组中的位置初始化
    shiftUp(count);
  }

  int extractMinIndex() {
    int retIndex = indexes.at(1);
    std::swap(indexes.at(1), indexes.at(count));
    reverse[indexes[1]] = 1;
    reverse[indexes[count]] = 0;
    count--;
    shiftDown(1);
    return retIndex;
  }

  Item extractMin() { return datas.at(extractMinIndex()); }

  bool contain(int i) {
    assert(i >= 0 && i + 1 <= indexes.size() - 1);
    return reverse[i + 1] != 0;
  }

  Item getItemByIndex(int i) {
    assert(contain(i));
    return datas.at(++i);
  }

  void change(int i, Item newItem) {
    assert(contain(i));
    datas[++i] = newItem;
    int k = reverse[i];
    shiftDown(k);
    shiftUp(k);
  }

private:
  void shiftUp(int k) {
    // 因为是数组实现的完全二叉树，所以子节点找父亲直接索引除以2即可
    while (k > 1 && datas[indexes[k]] < datas[indexes[k / 2]]) {
      std::swap(indexes[k], indexes[k / 2]);

      reverse[indexes[k / 2]] = k / 2;
      reverse[indexes[k]] = k;
      k /= 2;
    }
  }

  void shiftDown(int k) {
    while (k * 2 <= size()) {
      int j = 2 * k;

      if (j + 1 <= size() && datas[indexes[j + 1]] < datas[indexes[j]]) {
        j++;
      }

      if (datas[indexes[k]] <= datas[indexes[j]]) {
        break;
      }

      std::swap(indexes[k], indexes[j]);
      reverse[indexes[k]] = k;
      reverse[indexes[j]] = j;
      k = j;
    }
  }
};
} // namespace algo_and_ds::tree

#endif