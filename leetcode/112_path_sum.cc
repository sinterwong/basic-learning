/**
 * @file 112_path_sum.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the root of a binary tree and an integer targetSum, return true
if the tree has a root-to-leaf path such that adding up all the values along the
path equals targetSum.

A leaf is a node with no children.
 * @version 0.1
 * @date 2024-11-20
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "tree_helper.hpp"
#include <gtest/gtest.h>

using namespace std;
using namespace leetcode;

class Solution {
public:
  bool hasPathSum(TreeNode *root, int targetSum) {
    if (!root) {
      return false;
    }

    if (!root->left && !root->right) {
      return targetSum == root->val;
    }

    return hasPathSum(root->left, targetSum - root->val) ||
           hasPathSum(root->right, targetSum - root->val);
  }
};

TEST(PathSumTest, Normal) {
  Solution solution;
  vector<string> values1 = {"5", "4", "8",    "11",   "null", "13", "4",
                            "7", "2", "null", "null", "null", "1"};
  TreeNode *root1 = createTree(values1);
  printTree(root1);
  ASSERT_TRUE(solution.hasPathSum(root1, 22));
  deleteTree(root1);
  values1 = {"1", "2", "3"};
  root1 = createTree(values1);
  printTree(root1);
  ASSERT_FALSE(solution.hasPathSum(root1, 1));
  deleteTree(root1);
  ASSERT_FALSE(solution.hasPathSum(nullptr, 1));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
