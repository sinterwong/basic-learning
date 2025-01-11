/**
 * @file 83_remove_duplicates.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the head of a sorted linked list, delete all duplicates such
 * that each element appears only once. Return the linked list sorted as well.
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
  ListNode *deleteDuplicates(ListNode *head) {
    if (!head || !head->next) {
      return head;
    }
    ListNode *cur = head;

    while (cur && cur->next) {
      if (cur->val == cur->next->val) {
        auto delNode = cur->next;
        cur->next = cur->next->next;
        delete delNode;
      } else {
        cur = cur->next;
      }
    }
    return head;
  }
};

TEST(RemoveDuplicatesTest, Normal) {
  Solution s;
  ListNode *head = createLinkList({1, 1, 2, 3, 3});
  printLinkedList(head);
  ListNode *newHead = s.deleteDuplicates(head);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({1, 2, 3})));
  deleteLinkList(newHead);

  head = createLinkList({1, 1, 2});
  printLinkedList(head);
  newHead = s.deleteDuplicates(head);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({1, 2})));
  deleteLinkList(newHead);

  head = createLinkList({1, 1, 1});
  printLinkedList(head);
  newHead = s.deleteDuplicates(head);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({1})));
  deleteLinkList(newHead);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}