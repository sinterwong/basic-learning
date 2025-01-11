/**
 * @file 144_binary_tree_preorder_traversal.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
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
  vector<int> preorderTraversal(TreeNode *root) {
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
        if (cmd.node->right)
          s.push(Command{"go", cmd.node->right});
        if (cmd.node->left)
          s.push(Command{"go", cmd.node->left});
        s.push(Command{"print", cmd.node});
      } else {
        rets.push_back(cmd.node->val);
      }
    }
    return rets;
  }
};

TEST(BinaryTreePreorderTraversalTest, Normal) {
  Solution solution;

  vector<string> values1 = {"1", "null", "2", "3"};
  TreeNode *root1 = createTree(values1);
  printTree(root1);
  vector<int> expected1 = {1, 2, 3};
  EXPECT_EQ(solution.preorderTraversal(root1), expected1);
  deleteTree(root1);
  EXPECT_EQ(solution.preorderTraversal(nullptr), vector<int>());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
