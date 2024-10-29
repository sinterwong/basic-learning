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
#include <gtest/gtest.h>

using namespace std;

struct ListNode {
  int val;
  ListNode *next;
  ListNode() : val(0), next(nullptr) {}
  ListNode(int x) : val(x), next(nullptr) {}
  ListNode(int x, ListNode *next) : val(x), next(next) {}
};

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
  ASSERT_EQ(reversedHead->val, 5);
  ASSERT_EQ(reversedHead->next->val, 4);
  ASSERT_EQ(reversedHead->next->next->val, 3);
  ASSERT_EQ(reversedHead->next->next->next->val, 2);
  ASSERT_EQ(reversedHead->next->next->next->next->val, 1);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
