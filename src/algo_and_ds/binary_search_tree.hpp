/**
 * @file binary_search_tree.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * 二分搜索树是一个二叉树，主要作用是方便查询和维护动态的数据，天然有递归性质。
 * 定义： 左边的孩子小于自己，右边孩子大于自己
 * 优势：查找，增加，删除的时间复杂度都是O(logn)级别
 * @version 0.1
 * @date 2022-11-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <cassert>
#include <iostream>
#include <queue>

#ifndef __AADS_BINARY_SEARCH_TREE_HPP_
#define __AADS_BINARY_SEARCH_TREE_HPP_

namespace algo_and_ds::tree {

template <typename Key, typename Value> class BinarySearchTree {

  struct Node {
    Node *left;  // key < this
    Node *right; // key > this
    Key key;
    Value value;

    Node(Key _key, Value _value) : key(_key), value(_value) {
      left = right = nullptr;
    }

    Node(Node *node)
        : left(node->left), right(node->right), key(node->key),
          value(node->value) {}
  };

public:
  /**
   * @brief 插入
   *
   */
  void insert(Key key, Value value) { root = insert(root, key, value); }

  /**
   * @brief 搜索
   *
   */
  Value *search(Key key) { return search(root, key); }

  /**
   * @brief 是否包含该key
   *
   */
  bool contain(Key key) { return contain(root, key); }

  /**
   * @brief 深度优先遍历
   * 有前中后序遍历，每次遍历都是自己，左节点，右节点，只是根据前中后序来决定是否对当前节点进行操作
   * 用途：
   *   前序遍历：用于遍历整棵树就可以，比较简单
   *   中序遍历：可用于得到元素排序后的结果
   *   后续遍历：自底向上的遍历，多用于析构时从叶子节点开始释放
   */

  void preOrder() { preOrder(root); }

  void inOrder() { inOrder(root); }

  void postOrder() { postOrder(root); }

  void levelOrder() {
    std::queue<Node *> q;
    q.push(root);
    while (!q.empty()) {
      Node *curNode = q.front();
      q.pop();
      std::cout << curNode->key << std::endl;
      if (curNode->left != nullptr) {
        q.push(curNode->left);
      }
      if (curNode->right != nullptr) {
        q.push(curNode->right);
      }
    }
  }

  Key minimum() {
    assert(count != 0);
    Node *minNode = minimum(root);
    return minNode->key;
  }

  void removeMin() {
    if (root) {
      root = removeMin(root);
    }
  }

  void removeMax() {
    if (root) {
      root = removeMax(root);
    }
  }

  void remove(Key key) { root = remove(root, key); }

  bool isEmpty() { return count == 0; }

  int size() { return count; }

  ~BinarySearchTree() {
    // 使用后序遍历的方式析构整个二叉搜索树
    destroy(root);
  }

private:
  // 整树的根节点（各种操作的起点）
  Node *root = nullptr;

  // 记录树中节点的数量
  int count = 0;

private:
  Node *insert(Node *node, Key key, Value value) {

    // 递归结束条件之一，如果该节点没有值，就完成新增
    if (node == nullptr) {
      ++count;
      return new Node(key, value);
    }

    // 如果存在重复的key，就将新的值替换即可
    // 否则就向下递归（node->key > key 向左，node->key < key 向右）
    if (node->key == key) {
      node->value = value;
    } else if (node->key > key) {
      node->left = insert(node->left, key, value);
    } else {
      node->right = insert(node->right, key, value);
    }

    return node;
  }

  Value *search(Node *node, Key key) {
    if (node == nullptr) {
      return nullptr;
    }

    if (node->key == key) {
      return &(node->value);
    } else if (node->key > key) {
      return search(node->left, key);
    } else {
      return search(node->right, key);
    }
  }

  bool contain(Node *node, Key key) {
    if (node == nullptr) {
      return false;
    }
    if (node->key == key) {
      return true;
    } else if (node->key > key) {
      return contain(node->left, key);
    } else {
      return contain(node->right, key);
    }
  }

  void preOrder(Node *node) {
    if (node != nullptr) {
      std::cout << node->key << std::endl;
      preOrder(node->left);
      preOrder(node->right);
    }
  }

  void inOrder(Node *node) {
    if (node != nullptr) {
      inOrder(node->left);
      std::cout << node->key << std::endl;
      inOrder(node->right);
    }
  }

  void postOrder(Node *node) {
    if (node != nullptr) {
      postOrder(node->left);
      postOrder(node->right);
      std::cout << node->key << std::endl;
    }
  }

  void destroy(Node *node) {
    if (node != nullptr) {
      destroy(node->left);
      destroy(node->right);

      delete node;
      count--;
    }
  }

  Node *minimum(Node *node) {
    Node *ret = node;
    while (ret->left) {
      ret = ret->left;
    }
    return node;
  }

  Node *maximum(Node *node) {
    if (node->right == nullptr) {
      return node;
    }
    return maximum(node->right);
  }

  Node *removeMin(Node *node) {
    if (node->left == nullptr) {
      // 此时已经没有更小的节点了
      Node *newLeftNode = node->right;
      delete node;
      count--;
      return newLeftNode;
    }

    node->left = removeMin(node->left);
    return node;
  }

  Node *removeMax(Node *node) {
    if (node->right == nullptr) {
      // 此时已经没有更大的节点
      Node *newRightNode = node->left;
      delete node;
      count--;
      return newRightNode;
    }
    node->right = removeMax(node->right);
    return node;
  }

  Node *remove(Node *node, Key key) {
    if (node == nullptr) {
      return nullptr;
    }

    if (key < node->key) {
      node->left = remove(node->left, key);
      return node;
    } else if (key > node->key) {
      node->right = remove(node->right, key);
      return node;
    } else {
      if (node->left == nullptr) {
        Node *rightNode = node->right;
        delete node;
        count--;
        return rightNode;
      }

      if (node->right == nullptr) {
        Node *leftNode = node->left;
        delete node;
        count--;
        return leftNode;
      }

      Node *successor = new Node(minimum(node->right));
      count++;
      successor->right = removeMin(node->right);
      successor->left = node->left;

      delete node;
      count--;
      return successor;
    }
  }
};

} // namespace algo_and_ds::tree
#endif