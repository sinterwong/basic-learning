/**
 * @file 206_reverse_linked_list.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the head of a singly linked list, reverse the list, and return
 * the reversed list.
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
  ListNode *reverseList(ListNode *head) {
    ListNode *pre = nullptr;
    ListNode *cur = head;

    while (cur != nullptr) {
      auto next = cur->next;
      cur->next = pre;
      pre = cur;
      cur = next;
    }
    return pre;
  }
};

TEST(ReverseLinkedListTest, Normal) {
  Solution s;
  ListNode *head = new ListNode(
      1, new ListNode(
             2, new ListNode(3, new ListNode(4, new ListNode(5, nullptr)))));
  ListNode *reversedHead = s.reverseList(head);

  ASSERT_TRUE(
      compareTwoLinks(reversedHead, leetcode::createLinkList({5, 4, 3, 2, 1})));
  deleteLinkList(reversedHead);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}