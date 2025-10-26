/**
 * @file 226_invert_binary_tree.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the root of a binary tree, invert the tree, and return its root.
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
  TreeNode *invertTree(TreeNode *root) {
    if (!root) {
      return nullptr;
    }

    invertTree(root->left);
    invertTree(root->right);

    swap(root->left, root->right);

    return root;
  }
};

TEST(InvertBinaryTreeTest, Normal) {
  Solution solution;
  vector<string> values1 = {"4", "2", "7", "1", "3", "6", "9"};
  TreeNode *root1 = createTree(values1);
  printTree(root1);
  vector<string> expected1 = {"4", "7", "2", "9", "6", "3", "1"};
  TreeNode *expectedRoot = createTree(expected1);
  printTree(expectedRoot);
  ASSERT_TRUE(compareTwoTrees(solution.invertTree(root1), expectedRoot));
  deleteTree(root1);
  deleteTree(expectedRoot);
  ASSERT_TRUE(compareTwoTrees(solution.invertTree(nullptr), nullptr));

  values1 = {"2", "1", "3"};
  root1 = createTree(values1);
  printTree(root1);
  expected1 = {"2", "3", "1"};
  expectedRoot = createTree(expected1);
  printTree(expectedRoot);
  ASSERT_TRUE(compareTwoTrees(solution.invertTree(root1), expectedRoot));
  deleteTree(root1);
  deleteTree(expectedRoot);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
