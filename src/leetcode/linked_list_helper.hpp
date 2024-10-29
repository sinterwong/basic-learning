#include <iostream>
#include <vector>

#ifndef __LINKED_LIST_HELPER_HPP_
#define __LINKED_LIST_HELPER_HPP_

namespace leetcode {
struct ListNode {
  int val;
  ListNode *next;
  ListNode() : val(0), next(nullptr) {}
  ListNode(int x) : val(x), next(nullptr) {}
  ListNode(int x, ListNode *next) : val(x), next(next) {}
};

inline void printLinkedList(ListNode *head) {
  while (head != nullptr) {
    std::cout << head->val << " -> ";
    head = head->next;
  }
  std::cout << "null" << std::endl;
}

inline ListNode *createLinkList(std::vector<int> const &nums) {
  if (nums.empty()) {
    return nullptr;
  }

  ListNode *head = new ListNode(nums[0]);
  ListNode *cur = head;
  for (int i = 1; i < nums.size(); ++i) {
    cur->next = new ListNode(nums[i]);
    cur = cur->next;
  }
  return head;
}

inline void deleteLinkList(ListNode *head) {
  while (head != nullptr) {
    ListNode *next = head->next;
    delete head;
    head = next;
  }
}
} // namespace leetcode
#endif