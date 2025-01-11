/**
 * @file 19_remove_nth_node_from_end.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the head of a linked list, remove the nth node from the end of
 * the list and return its head.
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
  ListNode *removeNthFromEnd(ListNode *head, int n) {
    ListNode *dummy = new ListNode(0, head);

    ListNode *p = dummy;
    ListNode *q = dummy;
    for (int i = 0; i < n + 1; ++i) {
      q = q->next;
    }

    while (q) {
      p = p->next;
      q = q->next;
    }
    auto delNode = p->next;
    p->next = p->next->next;

    delete delNode;

    auto ret = dummy->next;
    delete dummy;
    return ret;
  }
};

TEST(RemoveNthNodeFromEndOfListTest, Normal) {
  Solution s;
  ListNode *head = createLinkList({1, 2, 3, 4, 5});
  printLinkedList(head);
  ListNode *newHead = s.removeNthFromEnd(head, 2);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({1, 2, 3, 5})));
  deleteLinkList(newHead);

  head = createLinkList({1});
  printLinkedList(head);
  newHead = s.removeNthFromEnd(head, 1);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({})));
  deleteLinkList(newHead);

  head = createLinkList({1, 2});
  printLinkedList(head);
  newHead = s.removeNthFromEnd(head, 1);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({1})));
  deleteLinkList(newHead);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
