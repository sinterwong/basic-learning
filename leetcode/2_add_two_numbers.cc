/**
 * @file 2_add_two_numbers.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief You are given two non-empty linked lists representing two non-negative
integers. The digits are stored in reverse order, and each of their nodes
contains a single digit. Add the two numbers and return the sum as a linked
list.

You may assume the two numbers do not contain any leading zero, except the
number 0 itself.
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
  ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {

    ListNode *outDummy = new ListNode(0);
    ListNode *out = outDummy;
    int carry = 0;
    while (l1 || l2 || carry) {
      int l1v = 0, l2v = 0;
      if (l1) {
        l1v = l1->val;
        l1 = l1->next;
      }
      if (l2) {
        l2v = l2->val;
        l2 = l2->next;
      }
      auto ov = l1v + l2v + carry;
      if (ov >= 10) {
        carry = 1;
        ov %= 10;
      } else {
        carry = 0;
      }
      out->next = new ListNode(ov);
      out = out->next;
    }
    return outDummy->next;
  }
};

TEST(AddTwoNumberTest, Normal) {
  Solution s;
  ListNode *l1 = createLinkList({2, 4, 3});
  ListNode *l2 = createLinkList({5, 6, 4});
  printLinkedList(l1);
  printLinkedList(l2);
  ListNode *l3 = s.addTwoNumbers(l1, l2);
  printLinkedList(l3);
  ASSERT_TRUE(compareTwoLinks(l3, createLinkList({7, 0, 8})));
  deleteLinkList(l3);

  l1 = createLinkList({0});
  l2 = createLinkList({0});
  l3 = s.addTwoNumbers(l1, l2);
  printLinkedList(l3);
  ASSERT_TRUE(compareTwoLinks(l3, createLinkList({0})));
  deleteLinkList(l3);

  l1 = createLinkList({9, 9, 9, 9, 9, 9, 9});
  l2 = createLinkList({9, 9, 9, 9});
  l3 = s.addTwoNumbers(l1, l2);
  printLinkedList(l3);
  ASSERT_TRUE(compareTwoLinks(l3, createLinkList({8, 9, 9, 9, 0, 0, 0, 1})));
  deleteLinkList(l3);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
