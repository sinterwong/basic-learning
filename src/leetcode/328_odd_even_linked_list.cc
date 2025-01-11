/**
 * @file 328_odd_even_linked_list.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the head of a singly linked list, group all the nodes with odd
indices together followed by the nodes with even indices, and return the
reordered list.

The first node is considered odd, and the second node is even, and so on.

Note that the relative order inside both the even and odd groups should remain
as it was in the input.

You must solve the problem in O(1) extra space complexity and O(n) time
complexity.
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
  ListNode *oddEvenList(ListNode *head) {
    if (!head || !head->next) {
      return head;
    }

    ListNode *oddHead = new ListNode(0);
    ListNode *evenHead = new ListNode(0);

    auto odd = oddHead;
    auto even = evenHead;

    int count = 1;
    while (head) {
      if (count % 2 != 0) {
        odd->next = head;
        odd = odd->next;
      } else {
        even->next = head;
        even = even->next;
      }
      head = head->next;
      count++;
    }
    even->next = nullptr;
    odd->next = evenHead->next;
    return oddHead->next;
  }
};

TEST(OddEvenLinkedListTest, Normal) {
  Solution s;
  ListNode *head = createLinkList({1, 2, 3, 4, 5});
  printLinkedList(head);
  ListNode *newHead = s.oddEvenList(head);
  printLinkedList(newHead);

  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({1, 3, 5, 2, 4})));

  deleteLinkList(newHead);

  head = createLinkList({2, 1, 3, 5, 6, 4, 7});
  printLinkedList(head);
  newHead = s.oddEvenList(head);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({2, 3, 6, 7, 1, 5, 4})));
  deleteLinkList(newHead);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
