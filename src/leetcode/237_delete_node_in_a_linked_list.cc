/**
 * @file 237_delete_node_in_a_linked_list.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief There is a singly-linked list head and we want to delete a node node
in it.

You are given the node to be deleted node. You will not be given access to the
first node of head.

All the values of the linked list are unique, and it is guaranteed that the
given node node is not the last node in the linked list.

Delete the given node. Note that by deleting the node, we do not mean removing
it from memory. We mean:

The value of the given node should not exist in the linked list.
The number of nodes in the linked list should decrease by one.
All the values before node should be in the same order.
All the values after node should be in the same order.
Custom testing:

For the input, you should provide the entire linked list head and the node to be
given node. node should not be the last node of the list and should be an actual
node in the list. We will build the linked list and pass the node to your
function. The output will be the entire list after calling your function.

 * @version 0.1
 * @date 2024-11-06
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "linked_list_helper.hpp"
#include <gtest/gtest.h>

using namespace std;
using namespace leetcode;

class Solution {
public:
  void deleteNode(ListNode *node) {
    if (!node) {
      return;
    }

    if (!node->next) {
      delete node;
      node = nullptr;
      return;
    }

    node->val = node->next->val;
    auto delNode = node->next;
    node->next = delNode->next;
    delete delNode;
  }
};

TEST(DeleteNodeInALinkedListTest, Normal) {
  Solution s;
  ListNode *head = createLinkList({4, 5, 1, 9});
  ListNode *node = head->next;
  s.deleteNode(node);
  ASSERT_TRUE(compareTwoLinks(head, createLinkList({4, 1, 9})));
  deleteLinkList(head);

  head = createLinkList({4, 5, 1, 9});
  node = head->next->next;
  s.deleteNode(node);
  ASSERT_TRUE(compareTwoLinks(head, createLinkList({4, 5, 9})));
  deleteLinkList(head);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
