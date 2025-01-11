/**
 * @file 203_remove_linked_list_elements.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the head of a linked list and an integer val, remove all the
 * nodes of the linked list that has Node.val == val, and return the new head.
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
  ListNode *removeElements(ListNode *head, int val) {
    ListNode *dummy = new ListNode(0, head);
    auto pre = dummy;
    while (pre->next) {
      if (pre->next->val == val) {
        pre->next = pre->next->next;
      } else {
        pre = pre->next;
      }
    }
    return dummy->next;
  }
};

TEST(RemoveLinkedListElementsTest, Normal) {
  Solution s;
  ListNode *head = createLinkList({1, 2, 6, 3, 4, 5, 6});
  printLinkedList(head);
  ListNode *newHead = s.removeElements(head, 6);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({1, 2, 3, 4, 5})));
  deleteLinkList(newHead);

  head = createLinkList({});
  printLinkedList(head);
  newHead = s.removeElements(head, 1);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({})));
  deleteLinkList(newHead);

  head = createLinkList({7, 7, 7, 7});
  printLinkedList(head);
  newHead = s.removeElements(head, 7);
  printLinkedList(newHead);
  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({})));
  deleteLinkList(newHead);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
