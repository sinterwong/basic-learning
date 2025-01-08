/**
 * @file index_max_heap.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-01-26
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __AADS_INDEX_MAX_HEAP_HPP_
#define __AADS_INDEX_MAX_HEAP_HPP_

#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

namespace algo_and_ds::tree {
template <typename Item> class IndexMaxHeap {
private:
  // 数据的配套索引，用于根据用户传入的索引从数据中查询数据
  std::vector<int> indexes;

  /*
   * 反向查找表性质：
   * indexes[i] = j
   * reverse[j] = i
   *
   * indexes[reverse[i]] = i
   * reverse[indexes[i]] = i
   */
  std::vector<int> reverse; // 表示数据的配套索引在堆数组中的位置

  // 堆中数据
  std::vector<Item> datas;

  // 当前堆中的数据量
  int count = 0;

  void shiftUp(size_t k) {
    while (k > 1 && datas.at(indexes.at(k)) > datas.at(indexes.at(k / 2))) {
      std::swap(indexes.at(k), indexes.at(k / 2));

      reverse[indexes[k / 2]] = k / 2;
      reverse[indexes[k]] = k;
      k /= 2;
    }
  }

  void shiftDown(size_t k) {
    // k所在的索引还存在左子树的情况下就要一直向下遍历
    while (2 * k <= size()) {
      int j = 2 * k;
      // 存在右子树且右子树的值比左子树大的情况
      if (j + 1 <= size() &&
          datas.at(indexes.at(j + 1)) > datas.at(indexes.at(j))) {
        // 更新比较子树为右子树
        j++;
      }

      if (datas.at(indexes.at(k)) >= datas.at(indexes.at(j))) {
        // 父节点此时已经大于最大子节点，shiftDown过程结束
        break;
      }

      std::swap(indexes.at(k), indexes.at(j));
      reverse[indexes[k]] = k;
      reverse[indexes[j]] = j;
      k = j;
    }
  }

public:
  IndexMaxHeap(int capacity) {
    datas.resize(capacity + 1);
    indexes.resize(capacity + 1);
    reverse.resize(capacity + 1);
  }

  size_t size() { return count; }

  bool isEmpty() { return count == 0; }

  void insert(int i, Item d) {
    assert(count + 1 <= indexes.size() - 1);
    assert(i >= 0 && i + 1 <= indexes.size() - 1);
    // 该数据结构维护的索引是从1开始的
    i += 1;
    datas[i] = d;
    indexes[++count] = i;
    reverse[i] = count; // i索引 在索引数组中的位置初始化
    shiftUp(count);
  }

  int extractMaxIndex() {
    int retIndex = indexes.at(1);
    std::swap(indexes.at(1), indexes.at(count));
    reverse[indexes[1]] = 1;
    reverse[indexes[count]] = 0; // 归位
    count--;
    shiftDown(1);
    return retIndex - 1;
  }

  Item extractMax() { return datas.at(extractMaxIndex() + 1); }

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
};
} // namespace algo_and_ds::tree
#endif