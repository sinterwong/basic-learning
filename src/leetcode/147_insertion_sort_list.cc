/**
 * @file 147_insertion_sort_list.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the head of a singly linked list, sort the list using insertion
sort, and return the sorted list's head.

The steps of the insertion sort algorithm:

Insertion sort iterates, consuming one input element each repetition and growing
a sorted output list. At each iteration, insertion sort removes one element from
the input data, finds the location it belongs within the sorted list and inserts
it there. It repeats until no input elements remain. The following is a
graphical example of the insertion sort algorithm. The partially sorted list
(black) initially contains only the first element in the list. One element (red)
is removed from the input data and inserted in-place into the sorted list with
each iteration.
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
  ListNode *insertionSortList(ListNode *head) {
    ListNode *dummy = new ListNode(0);
    ListNode *cur = head;
    while (cur) {
      ListNode *next = cur->next;
      ListNode *pre = dummy;
      while (pre->next && pre->next->val < cur->val) {
        pre = pre->next;
      }
      cur->next = pre->next;
      pre->next = cur;
      cur = next;
    }
    return dummy->next;
  }
};

TEST(InsertionSortListTest, Normal) {
  Solution s;
  ListNode *head = createLinkList({4, 2, 1, 3});
  printLinkedList(head);
  ListNode *newHead = s.insertionSortList(head);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({1, 2, 3, 4})));
  deleteLinkList(newHead);

  head = createLinkList({-1, 5, 3, 4, 0});
  printLinkedList(head);
  newHead = s.insertionSortList(head);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({-1, 0, 3, 4, 5})));
  deleteLinkList(newHead);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
