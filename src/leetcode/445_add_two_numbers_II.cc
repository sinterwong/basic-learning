/**
 * @file 445_add_two_numbers_II.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief You are given two non-empty linked lists representing two non-negative
integers. The most significant digit comes first and each of their nodes
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
#include <stack>

using namespace std;
using namespace leetcode;

class Solution {
public:
  ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
    stack<int> l1vs;
    stack<int> l2vs;
    while (l1) {
      l1vs.push(l1->val);
      l1 = l1->next;
    }

    while (l2) {
      l2vs.push(l2->val);
      l2 = l2->next;
    }

    stack<int> output;

    int carry = 0;
    while (!l1vs.empty() || !l2vs.empty() || carry) {
      int l1v = 0, l2v = 0;
      if (!l1vs.empty()) {
        l1v = l1vs.top();
        l1vs.pop();
      }
      if (!l2vs.empty()) {
        l2v = l2vs.top();
        l2vs.pop();
      }
      auto ov = l1v + l2v + carry;
      if (ov >= 10) {
        carry = 1;
        ov %= 10;
      } else {
        carry = 0;
      }
      output.push(ov);
    }
    ListNode *outDummy = new ListNode(0);
    ListNode *out = outDummy;

    while (!output.empty()) {
      out->next = new ListNode(output.top());
      output.pop();
      out = out->next;
    }
    return outDummy->next;
  }
};

TEST(AddTwoNumbersIITest, Normal) {
  Solution s;
  ListNode *l1 = createLinkList({7, 2, 4, 3});
  ListNode *l2 = createLinkList({5, 6, 4});
  printLinkedList(l1);
  printLinkedList(l2);
  ListNode *l3 = s.addTwoNumbers(l1, l2);
  printLinkedList(l3);
  ASSERT_TRUE(compareTwoLinks(l3, createLinkList({7, 8, 0, 7})));
  deleteLinkList(l3);

  l1 = createLinkList({2, 4, 3});
  l2 = createLinkList({5, 6, 4});
  l3 = s.addTwoNumbers(l1, l2);
  printLinkedList(l3);
  ASSERT_TRUE(compareTwoLinks(l3, createLinkList({8, 0, 7})));
  deleteLinkList(l3);

  l1 = createLinkList({0});
  l2 = createLinkList({0});
  l3 = s.addTwoNumbers(l1, l2);
  printLinkedList(l3);
  ASSERT_TRUE(compareTwoLinks(l3, createLinkList({0})));
  deleteLinkList(l3);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
