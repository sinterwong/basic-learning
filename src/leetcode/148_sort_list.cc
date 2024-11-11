/**
 * @file 148_sort_list.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the head of a linked list, return the list after sorting it in
 * ascending order.
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
  ListNode *sortList(ListNode *head) {
    if (!head || !head->next) {
      return head;
    }

    ListNode *slow = head;
    ListNode *fast = head->next;

    while (fast && fast->next) {
      slow = slow->next;
      fast = fast->next->next;
    }

    ListNode *mid = slow->next;

    // split the list into two halves
    slow->next = nullptr;

    ListNode *left = sortList(head);
    ListNode *right = sortList(mid);

    return merge(left, right);
  }

private:
  ListNode *merge(ListNode *l1, ListNode *l2) {
    ListNode *dummy = new ListNode(0);
    ListNode *tail = dummy;

    while (l1 || l2) {
      if (!l1) {
        tail->next = l2;
        l2 = l2->next;
      } else if (!l2) {
        tail->next = l1;
        l1 = l1->next;
      } else if (l1->val < l2->val) {
        tail->next = l1;
        l1 = l1->next;
      } else {
        tail->next = l2;
        l2 = l2->next;
      }
      tail = tail->next;
    }
    return dummy->next;
  }
};

TEST(SortListTest, Normal) {
  Solution s;
  ListNode *head = createLinkList({4, 2, 1, 3});
  printLinkedList(head);
  ListNode *newHead = s.sortList(head);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({1, 2, 3, 4})));
  deleteLinkList(newHead);

  head = createLinkList({-1, 5, 3, 4, 0});
  printLinkedList(head);
  newHead = s.sortList(head);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({-1, 0, 3, 4, 5})));
  deleteLinkList(newHead);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
