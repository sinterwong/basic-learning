/**
 * @file 25_reverse_nodes_in_kGroup.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the head of a linked list, reverse the nodes of the list k at a
time, and return the modified list.

k is a positive integer and is less than or equal to the length of the linked
list. If the number of nodes is not a multiple of k then left-out nodes, in the
end, should remain as it is.

You may not alter the values in the list's nodes, only nodes themselves may be
changed.
 * @version 0.1
 * @date 2024-11-07
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
  ListNode *reverseKGroup(ListNode *head, int k) {
    if (!head || k == 1)
      return head;

    ListNode *dummy = new ListNode(0, head);
    ListNode *pre = dummy;
    ListNode *cur = head;

    while (true) {
      ListNode *tail = pre;
      for (int i = 0; i < k; ++i) {
        tail = tail->next;
        if (!tail)
          return dummy->next;
      }

      ListNode *ngh = tail->next;

      // reverse
      ListNode *prev = ngh;
      ListNode *cgh = cur;
      for (int i = 0; i < k; ++i) {
        ListNode *next = cur->next;
        cur->next = prev;
        prev = cur;
        cur = next;
      }

      pre->next = prev;
      pre = cgh;
      cur = ngh;
    }

    return dummy->next;
  }
};

TEST(ReverseNodesInKGroupTest, Normal) {
  Solution s;
  ListNode *head = createLinkList({1, 2, 3, 4, 5});
  printLinkedList(head);
  ListNode *newHead = s.reverseKGroup(head, 2);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({2, 1, 4, 3, 5})));
  deleteLinkList(newHead);

  head = createLinkList({1, 2, 3, 4, 5});
  printLinkedList(head);
  newHead = s.reverseKGroup(head, 3);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({3, 2, 1, 4, 5})));
  deleteLinkList(newHead);

  head = createLinkList({});
  printLinkedList(head);
  newHead = s.reverseKGroup(head, 3);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({})));
  deleteLinkList(newHead);

  head = createLinkList({1});
  printLinkedList(head);
  newHead = s.reverseKGroup(head, 3);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({1})));
  deleteLinkList(newHead);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
