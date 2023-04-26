/**
 * @file binary_search_tree.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * 二分搜索树是一个二叉树，主要作用是方便查询和维护动态的数据，天然有递归性质。
 * 定义： 左边的孩子小于自己，右边孩子大于自己
 * 优势：查找，增加，删除的时间复杂度都是O(logn)级别
 * @version 0.1
 * @date 2022-11-20
 *
 * @copyright Copyright (c) 2022
 *
 */

namespace algo_and_ds::tree {

class BinarySearchTree {

  /**
   * @brief 插入
   *
   */

  /**
   * @brief 搜索
   *
   */

  /**
   * @brief
   * 遍历有前中后序遍历，每次遍历都是自己，左节点，右节点，只是根据前中后序来决定是否操作
   * 用途：
   *   前序遍历：用于遍历整棵树就可以，比较简单
   *   中序遍历：根据二叉树的定义，元素的顺序是排好序的
   *   后续遍历：自底向上的遍历，多用于析构时从叶子节点开始释放
   */
};

} // namespace algo_and_ds::tree