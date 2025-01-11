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

// inline void printTree(TreeNode *root) {
//   if (root == nullptr) {
//     cout << "#" << endl;
//     return;
//   }

//   queue<TreeNode *> q;
//   q.push(root);

//   while (!q.empty()) {
//     int levelSize = q.size();
//     for (int i = 0; i < levelSize; ++i) {
//       TreeNode *node = q.front();
//       q.pop();

//       if (node == nullptr) {
//         cout << "# ";
//       } else {
//         cout << node->val << " ";
//         q.push(node->left);
//         q.push(node->right);
//       }
//     }
//     cout << endl;
//   }
//   cout << endl;
// }

inline void printTree(TreeNode *root) {
  cout << "----------------------------" << endl;
  if (root == nullptr) {
    cout << " " << endl;
    return;
  }

  // compute the hight of the tree
  int height = 0;
  queue<TreeNode *> q;
  q.push(root);
  {
    queue<TreeNode *> temp = q;
    while (!temp.empty()) {
      int levelSize = temp.size();
      bool hasNode = false;

      for (int i = 0; i < levelSize; ++i) {
        TreeNode *node = temp.front();
        temp.pop();

        if (node != nullptr) {
          hasNode = true;
          temp.push(node->left);
          temp.push(node->right);
        } else {
          temp.push(nullptr);
          temp.push(nullptr);
        }
      }

      if (!hasNode)
        break;
      height++;
    }
  }

  int level = 0;
  while (!q.empty() && level < height) {
    int levelSize = q.size();
    int maxWidth = (1 << height) * 2;
    int nodeSpacing = (maxWidth / levelSize);
    string line;

    for (int i = 0; i < levelSize; ++i) {
      TreeNode *node = q.front();
      q.pop();

      int position = i * nodeSpacing + nodeSpacing / 2;

      while (line.length() < position) {
        line += " ";
      }

      if (node == nullptr) {
        line += " ";
        q.push(nullptr);
        q.push(nullptr);
      } else {
        string nodeStr = to_string(node->val);
        line += nodeStr;
        q.push(node->left);
        q.push(node->right);
      }
    }

    cout << line << endl;
    level++;
  }
  cout << "----------------------------" << endl;
}

inline bool compareTwoTrees(TreeNode *t1, TreeNode *t2) {
  if (t1 == nullptr && t2 == nullptr) {
    return true;
  }

  if (t1 == nullptr || t2 == nullptr) {
    return false;
  }

  if (t1->val != t2->val) {
    return false;
  }

  return compareTwoTrees(t1->left, t2->left) &&
         compareTwoTrees(t1->right, t2->right);
}
} // namespace leetcode
#endif