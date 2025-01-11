/**
 * @file 82_remove_duplicates_sorted_list_II.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the head of a sorted linked list, delete all nodes that have
 * duplicate numbers, leaving only distinct numbers from the original list.
 * Return the linked list sorted as well.
 * @version 0.1
 * @date 2024-11-05
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
    ListNode *dummy = new ListNode(0, head);
    ListNode *pre = dummy;

    while (head) {
      if (head->next && head->val == head->next->val) {
        while (head->next && head->val == head->next->val) {
          head = head->next;
        }
        pre->next = head->next;
      } else {
        pre = pre->next;
      }
      head = head->next;
    }
    return dummy->next;
  }
};

TEST(RemoveDuplicatesFromSortedListIITest, Normal) {
  Solution s;
  ListNode *head = createLinkList({1, 2, 3, 3, 4, 4, 5});
  printLinkedList(head);
  ListNode *newHead = s.deleteDuplicates(head);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({1, 2, 5})));
  deleteLinkList(newHead);

  head = createLinkList({1, 1, 1, 2, 3});
  printLinkedList(head);
  newHead = s.deleteDuplicates(head);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({2, 3})));
  deleteLinkList(newHead);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
