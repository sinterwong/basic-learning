#ifndef __TREE_HELPER_HELPER_HPP_
#define __TREE_HELPER_HELPER_HPP_
#include <cmath>
#include <iomanip>
#include <iostream>
#include <queue>

using namespace std;

namespace leetcode {
struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right)
      : val(x), left(left), right(right) {}
};

inline TreeNode *createTree(const vector<string> &values) {
  if (values.empty())
    return nullptr;

  std::queue<TreeNode *> q;
  TreeNode *root = new TreeNode(std::stoi(values[0]));
  q.push(root);

  for (size_t i = 1; i < values.size(); i += 2) {
    TreeNode *current = q.front();
    q.pop();

    // 处理左子节点
    if (i < values.size() && values[i] != "null") {
      current->left = new TreeNode(std::stoi(values[i]));
      q.push(current->left);
    }

    // 处理右子节点
    if (i + 1 < values.size() && values[i + 1] != "null") {
      current->right = new TreeNode(std::stoi(values[i + 1]));
      q.push(current->right);
    }
  }

  return root;
}

inline void deleteTree(TreeNode *root) {
  if (!root)
    return;
  deleteTree(root->left);
  deleteTree(root->right);
  delete root;
}

inline void printTree(TreeNode *root) {
  if (root == nullptr) {
    return;
  }
  queue<TreeNode *> q;
  q.push(root);
  while (!q.empty()) {
    int levelSize = q.size();
    for (int i = 0; i < levelSize; ++i) {
      TreeNode *node = q.front();
      q.pop();
      cout << node->val << " ";
      if (node->left) {
        q.push(node->left);
      }
      if (node->right) {
        q.push(node->right);
      }
    }
    cout << endl;
  }
}
} // namespace leetcode
#endif