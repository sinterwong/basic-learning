/**
 * @file test_binary_search_tree.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-01-30
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <iostream>
#include <string>

#include "binary_search_tree.hpp"

using namespace algo_and_ds::tree;

int main() {
  BinarySearchTree<int, std::string> bst;

  std::cout << "Inserting elements..." << std::endl;
  bst.insert(10, "ten");
  bst.insert(20, "twenty");
  bst.insert(5, "five");
  bst.insert(15, "fifteen");
  bst.insert(25, "twenty five");
  bst.insert(3, "three");
  bst.insert(7, "seven");

  std::cout << "Searching for 10: ";
  if (bst.search(10)) {
    std::cout << *bst.search(10) << std::endl;
  } else {
    std::cout << "not found." << std::endl;
  }

  std::cout << "Does the tree contain 20? " << bst.contain(20) << std::endl;

  std::cout << "PreOrder: " << std::endl;
  bst.preOrder();

  std::cout << "InOrder (sorted): " << std::endl;
  bst.inOrder();

  std::cout << "PostOrder: " << std::endl;
  bst.postOrder();

  std::cout << "LevelOrder: " << std::endl;
  bst.levelOrder();

  std::cout << "Minimum key: " << bst.minimum() << std::endl;

  std::cout << "Removing minimum..." << std::endl;
  bst.removeMin();
  std::cout << "New minimum: " << bst.minimum() << std::endl;

  std::cout << "Removing 15..." << std::endl;
  bst.remove(15);
  std::cout << "InOrder after removing 15: " << std::endl;
  bst.inOrder();

  std::cout << "Tree size: " << bst.size() << std::endl;

  std::cout << "Is tree empty? " << bst.isEmpty() << std::endl;

  return 0;
}