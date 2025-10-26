/**
 * @file 102_binary_tree_level_order_traversal.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the root of a binary tree, return the level order traversal of
 * its nodes' values. (i.e., from left to right, level by level).
 * @version 0.1
 * @date 2024-11-13
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "tree_helper.hpp"
#include <gtest/gtest.h>
#include <queue>

using namespace std;
using namespace leetcode;

class Solution {
public:
  vector<vector<int>> levelOrder(TreeNode *root) {
    vector<vector<int>> rets;
    if (!root) {
      return rets;
    }

    queue<pair<TreeNode *, int>> q;
    q.push({root, 0});

    while (!q.empty()) {
      auto [tn, level] = q.front();
      q.pop();

      if (level == rets.size()) {
        rets.push_back(vector<int>());
      }

      rets[level].push_back(tn->val);

      if (tn->left) {
        q.push({tn->left, level + 1});
      }

      if (tn->right) {
        q.push({tn->right, level + 1});
      }
    }
    return rets;
  }
};

TEST(BinaryTreeLevelOrderTraversalTest, Normal) {
  Solution solution;
  vector<string> values1 = {"3", "9", "20", "null", "null", "15", "7"};
  TreeNode *root1 = createTree(values1);
  printTree(root1);
  vector<vector<int>> expected1 = {{3}, {9, 20}, {15, 7}};
  EXPECT_EQ(solution.levelOrder(root1), expected1);
  deleteTree(root1);
  EXPECT_EQ(solution.levelOrder(nullptr), vector<vector<int>>());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
