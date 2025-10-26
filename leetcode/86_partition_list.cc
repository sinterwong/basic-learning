/**
 * @file 86_partition_list.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given the head of a linked list and a value x, partition it such that
all nodes less than x come before nodes greater than or equal to x.

You should preserve the original relative order of the nodes in each of the two
partitions.
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
  ListNode *partition(ListNode *head, int x) {
    ListNode *beforeHead = new ListNode(0);
    ListNode *afterHead = new ListNode(0);

    auto before = beforeHead;
    auto after = afterHead;

    while (head) {
      if (head->val < x) {
        before->next = head;
        before = before->next;
      } else {
        after->next = head;
        after = after->next;
      }
      head = head->next;
    }
    after->next = nullptr;
    before->next = afterHead->next;
    return beforeHead->next;
  }
};

TEST(PartitionListTest, Normal) {
  Solution s;
  ListNode *head = createLinkList({1, 4, 3, 2, 5, 2});
  printLinkedList(head);
  ListNode *newHead = s.partition(head, 3);
  printLinkedList(newHead);

  ASSERT_TRUE(compareTwoLinks(newHead, createLinkList({1, 2, 2, 4, 3, 5})));

  deleteLinkList(newHead);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
