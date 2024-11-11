/**
 * @file 24_swap_nodes_in_pairs.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given a linked list, swap every two adjacent nodes and return its
 * head. You must solve the problem without modifying the values in the list's
 * nodes (i.e., only nodes themselves may be changed.)
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
  ListNode *swapPairs(ListNode *head) {
    ListNode *dummy = new ListNode(0, head);
    ListNode *pre = dummy;

    while (pre->next && pre->next->next) {
      ListNode *n1 = pre->next;
      ListNode *n2 = n1->next;
      ListNode *next = n2->next;

      n2->next = n1;
      n1->next = next;
      pre->next = n2;

      pre = n1;
    }

    return dummy->next;
  }
};

TEST(SwapNodesInPairsTest, Normal) {
  Solution s;
  ListNode *head = createLinkList({1, 2, 3, 4});
  printLinkedList(head);
  ListNode *newHead = s.swapPairs(head);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({2, 1, 4, 3})));
  deleteLinkList(newHead);

  head = createLinkList({});
  printLinkedList(head);
  newHead = s.swapPairs(head);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({})));
  deleteLinkList(newHead);

  head = createLinkList({1});
  printLinkedList(head);
  newHead = s.swapPairs(head);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({1})));
  deleteLinkList(newHead);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
