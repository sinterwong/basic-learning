/**
 * @file 104_max_depth.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the root of a binary tree, return its maximum depth.
A binary tree's maximum depth is the number of nodes along the longest path from
the root node down to the farthest leaf node.
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
  int maxDepth(TreeNode *root) {
    if (!root) {
      return 0;
    }
    return max(maxDepth(root->left), maxDepth(root->right)) + 1;
  }
};

TEST(MaximumDepthOfBinaryTreeTest, Normal) {
  Solution solution;
  vector<string> values1 = {"3", "9", "20", "null", "null", "15", "7"};
  TreeNode *root1 = createTree(values1);
  ASSERT_EQ(solution.maxDepth(root1), 3);
  deleteTree(root1);
  ASSERT_EQ(solution.maxDepth(nullptr), 0);

  vector<string> values2 = {"1", "2"};
  root1 = createTree(values2);
  ASSERT_EQ(solution.maxDepth(root1), 2);
  deleteTree(root1);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
