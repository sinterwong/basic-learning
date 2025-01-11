/**
 * @file 49_group_anagrams.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Given an array of strings strs, group the anagrams together. You can
 * return the answer in any order.
 * @version 0.1
 * @date 2024-10-23
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <gtest/gtest.h>

using namespace std;

class Solution {
public:
  vector<vector<string>> groupAnagrams(vector<string> &strs) {
    map<array<int, 26>, vector<int>> sCountToIndexes;
    for (int i = 0; i < strs.size(); ++i) {
      array<int, 26> sCount = {0};
      for (auto c : strs[i]) {
        sCount[c - 'a']++;
      }
      sCountToIndexes[sCount].push_back(i);
    }

    vector<vector<string>> rets;
    for (auto const &p : sCountToIndexes) {
      vector<string> group;
      for (auto i : p.second) {
        group.push_back(strs[i]);
      }
      rets.push_back(group);
    }
    return rets;
  }
};

TEST(GroupAnagramsTest, Normal) {
  vector<string> strs = {"eat", "tea", "tan", "ate", "nat", "bat"};
  auto ret = Solution().groupAnagrams(strs);
  ASSERT_EQ(ret, vector<vector<string>>(
                     {{"tan", "nat"}, {"eat", "tea", "ate"}, {"bat"}}));

  strs = {""};
  ret = Solution().groupAnagrams(strs);
  ASSERT_EQ(ret, vector<vector<string>>({{""}}));

  strs = {"a"};
  ret = Solution().groupAnagrams(strs);
  ASSERT_EQ(ret, vector<vector<string>>({{"a"}}));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
