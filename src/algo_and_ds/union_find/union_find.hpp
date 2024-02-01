/**
 * @file union_find.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-01-31
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __AADS_UNION_FIND_HPP_
#define __AADS_UNION_FIND_HPP_
#include <cassert>
#include <vector>
namespace algo_and_ds::tree {

class UnionFindGroup {
private:
  std::vector<int> ids;

public:
  UnionFindGroup(int count) {
    ids.resize(count);
    for (int i = 0; i < count; ++i) {
      ids.at(i) = i;
    }
  }

  int find(int q) { return ids.at(q); }

  bool isConnection(int p, int q) { return find(p) == find(q); }

  void unionElements(int p, int q) {
    int pId = ids.at(p);
    int qId = ids.at(q);
    if (pId == qId) {
      return;
    }
    for (int i = 0; i < ids.size(); ++i) {
      if (ids[i] == pId) {
        ids[i] = qId;
      }
    }
  }
};

class UnionFindTreeBase {
private:
  std::vector<int> parent;

public:
  UnionFindTreeBase(int count) {
    parent.resize(count);
    for (int i = 0; i < count; ++i) {
      parent.at(i) = i;
    }
  }

  int find(int p) {
    // 一个节点一个节点的向上遍历直到根部（根部的定义是自己的根是自己）
    while (p != parent.at(p)) {
      p = parent.at(p);
    }
    return p;
  }

  bool isConnection(int p, int q) { return find(p) == find(q); }

  void unionElements(int p, int q) {
    int pRoot = find(p);
    int qRoot = find(q);

    if (pRoot == qRoot) {
      return;
    }

    // 将其中一个根变成另一个的根
    parent[pRoot] = qRoot;
  }
};

class UnionFindTreeOpSize {
private:
  std::vector<int> parent;
  std::vector<int> sz; // sz[i]表示以i为根的集合中的元素个数

public:
  UnionFindTreeOpSize(int count) {
    parent.resize(count);
    sz.resize(count);
    for (int i = 0; i < count; ++i) {
      parent[i] = i;
      sz[i] = 1;
    }
  }

  int find(int p) {
    // 一个节点一个节点的向上遍历直到根部（根部的定义是自己的根是自己）
    while (p != parent.at(p)) {
      // 路径压缩，如果p的父节点不是根节点，就换成更上一层的父亲
      parent[p] = parent[parent[p]];
      p = parent.at(p);
    }
    return p;
  }

  bool isConnection(int p, int q) { return find(p) == find(q); }

  void unionElements(int p, int q) {
    int pRoot = find(p);
    int qRoot = find(q);

    if (pRoot == qRoot) {
      return;
    }

    if (sz[pRoot] < sz[qRoot]) {
      // 将短根放到长根下面
      parent[pRoot] = qRoot;
      sz[qRoot] += sz[pRoot];
    } else {
      parent[qRoot] = pRoot;
      sz[pRoot] += sz[qRoot];
    }
  }
};

class UnionFindTreeOpRank {
private:
  std::vector<int> parent;
  std::vector<int> rank; // rank[i]表示以i为根的集合所表示的树的层数

public:
  UnionFindTreeOpRank(int count) {
    parent.resize(count);
    rank.resize(count);
    for (int i = 0; i < count; ++i) {
      parent[i] = i;
      rank[i] = 1;
    }
  }

  int find(int p) {
    // 递归的方式完成路径压缩的极端实现（让所有的子节点到父节点的距离只有1）
    if (p != parent[p]) {
      parent[p] = find(parent[p]);
    }
    // 直到找到根节点后，才一路返回根节点给每一个子节点更新父亲为根
    return parent[p];
  }

  bool isConnection(int p, int q) { return find(p) == find(q); }

  void unionElements(int p, int q) {
    int pRoot = find(p);
    int qRoot = find(q);

    if (pRoot == qRoot) {
      return;
    }

    if (rank[pRoot] < rank[qRoot]) {
      // 将短根放到长根下面，此时用的还是长根的rank，因此不需要维护
      parent[pRoot] = qRoot;
    } else if (rank[pRoot] > rank[qRoot]) {
      parent[qRoot] = pRoot;
    } else {
      // 随便把一个的根换成另一个，然后最终为根的rank层数增长1
      parent[pRoot] = qRoot;
      rank[qRoot]++;
    }
  }
};

} // namespace algo_and_ds::tree

#endif