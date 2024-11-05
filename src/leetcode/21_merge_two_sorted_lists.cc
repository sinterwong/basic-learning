/**
 * @file 21_merge_two_sorted_lists.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
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
  ListNode *mergeTwoLists(ListNode *list1, ListNode *list2) {
    ListNode *l1Dummy = new ListNode(0, list1);
    ListNode *l2Dummy = new ListNode(0, list2);
    ListNode *outputDummy = new ListNode(0);

    auto l1Pre = l1Dummy;
    auto l2Pre = l2Dummy;
    auto outPre = outputDummy;

    while (l1Pre->next || l2Pre->next) {
      if (!l1Pre->next) {
        outPre->next = new ListNode(l2Pre->next->val);
        l2Pre = l2Pre->next;
      } else if (!l2Pre->next) {
        outPre->next = new ListNode(l1Pre->next->val);
        l1Pre = l1Pre->next;
      } else if (l1Pre->next->val < l2Pre->next->val) {
        outPre->next = new ListNode(l1Pre->next->val);
        l1Pre = l1Pre->next;
      } else {
        outPre->next = new ListNode(l2Pre->next->val);
        l2Pre = l2Pre->next;
      }
      outPre = outPre->next;
    }
    return outputDummy->next;
  }
};

TEST(MergeTwoSortedListsTest, Normal) {
  Solution s;
  ListNode *l1 = createLinkList({1, 2, 4});
  ListNode *l2 = createLinkList({1, 3, 4});
  printLinkedList(l1);
  printLinkedList(l2);
  ListNode *l3 = s.mergeTwoLists(l1, l2);
  printLinkedList(l3);
  ASSERT_TRUE(compareTwoLinks(l3, createLinkList({1, 1, 2, 3, 4, 4})));
  deleteLinkList(l3);

  l1 = createLinkList({});
  l2 = createLinkList({});
  l3 = s.mergeTwoLists(l1, l2);
  printLinkedList(l3);
  ASSERT_TRUE(compareTwoLinks(l3, createLinkList({})));
  deleteLinkList(l3);

  l1 = createLinkList({});
  l2 = createLinkList({0});
  l3 = s.mergeTwoLists(l1, l2);
  printLinkedList(l3);
  ASSERT_TRUE(compareTwoLinks(l3, createLinkList({0})));
  deleteLinkList(l3);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
