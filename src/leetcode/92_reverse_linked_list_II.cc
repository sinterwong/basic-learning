/**
 * @file 92_reverse_linked_list_II.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the head of a singly linked list and two integers left and right
 * where left <= right, reverse the nodes of the list from position left to
 * position right, and return the reversed list.
 * @version 0.1
 * @date 2024-10-29
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
  ListNode *reverseBetween(ListNode *head, int left, int right) {
    if (!head || !head->next || left == right) {
      return head;
    }
    ListNode *dummy = new ListNode(0);
    dummy->next = head;
    ListNode *pre = dummy;

    for (int i = 0; i < left - 1; ++i) {
      pre = pre->next;
    }

    ListNode *cur = pre->next;

    for (int i = 0; i < right - left; ++i) {
      ListNode *next = cur->next;
      cur->next = next->next;
      next->next = pre->next;
      pre->next = next;
    }
    return dummy->next;
  }
};

TEST(ReverseLinkedListIITest, Normal) {
  Solution s;
  ListNode *head = createLinkList({1, 2, 3, 4, 5, 6});
  printLinkedList(head);
  ListNode *reversedHead = s.reverseBetween(head, 2, 5);
  printLinkedList(reversedHead);

  ASSERT_TRUE(
      compareTwoLinks(reversedHead, createLinkList({1, 5, 4, 3, 2, 6})));

  deleteLinkList(reversedHead);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}