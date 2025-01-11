/**
 * @file 145_binary_tree_postorder_traversal.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the root of a binary tree, return the postorder traversal of its
 * nodes' values.
 * @version 0.1
 * @date 2024-11-11
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "tree_helper.hpp"
#include <gtest/gtest.h>
#include <stack>

using namespace std;
using namespace leetcode;

struct Command {
  string cmd;
  TreeNode *node;

  Command(string const &cmd, TreeNode *node) : cmd(cmd), node(node) {}
};

class Solution {
public:
  vector<int> postorderTraversal(TreeNode *root) {
    vector<int> rets;

    if (!root) {
      return rets;
    }

    stack<Command> s;
    s.push(Command{"go", root});
    while (!s.empty()) {
      auto cmd = s.top();
      s.pop();
      if (cmd.cmd == "go") {
        s.push(Command{"print", cmd.node});
        if (cmd.node->right)
          s.push(Command{"go", cmd.node->right});
        if (cmd.node->left)
          s.push(Command{"go", cmd.node->left});
      } else {
        rets.push_back(cmd.node->val);
      }
    }
    return rets;
  }
};

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
