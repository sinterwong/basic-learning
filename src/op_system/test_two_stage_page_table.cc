#include "two_stage_page_table.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace op_system::virtual_mem;

class PageTableTest : public ::testing::Test {
protected:
  void SetUp() override {
    // 1GB 的物理内存空间 (使用页数表示)
    constexpr size_t ONE_GB = 1024 * 1024 * 1024;
    constexpr size_t TOTAL_PAGES = ONE_GB / PAGE_SIZE;
    paging_manager = std::make_unique<PagingManager>(TOTAL_PAGES);
  }

  std::unique_ptr<PagingManager> paging_manager;
};

TEST_F(PageTableTest, BasicMapping) {
  // 映射虚拟页面 0xF (pfn) 到物理页面 0x100 (pfn)
  uint32_t virtual_address = 0x0000F000; // 第15个虚拟页面
  uint32_t physical_frame = 0x100;       // 第256个物理页面

  ASSERT_TRUE(paging_manager->map_page(virtual_address, physical_frame));

  // 验证地址转换
  uint32_t translated_address =
      paging_manager->translate_virtual_to_physical(0x0000F123);
  ASSERT_EQ(translated_address, 0x00100123); // 0x100 * 0x1000 + 0x123
}

TEST_F(PageTableTest, PageFault) {
  // 尝试访问未映射的地址应该返回无效地址
  uint32_t virtual_address = 0x00001000;
  uint32_t physical_address =
      paging_manager->translate_virtual_to_physical(virtual_address);
  ASSERT_EQ(physical_address, ~0U); // 应该返回INVALID_ADDRESS (0xFFFFFFFF)
}

TEST_F(PageTableTest, MultiplePages) {
  // 映射多个虚拟页面到不同的物理页面
  const std::vector<std::pair<uint32_t, uint32_t>> mappings = {
      {0x00001000, 0x100}, // 映射第1个虚拟页面到第256个物理页面
      {0x00002000, 0x200}, // 映射第2个虚拟页面到第512个物理页面
      {0x00003000, 0x300}  // 映射第3个虚拟页面到第768个物理页面
  };

  // 创建映射
  for (const auto &[vaddr, pframe] : mappings) {
    ASSERT_TRUE(paging_manager->map_page(vaddr, pframe));
  }

  // 验证所有映射
  for (const auto &[vaddr, pframe] : mappings) {
    uint32_t expected_paddr = pframe * PAGE_SIZE;
    uint32_t translated = paging_manager->translate_virtual_to_physical(vaddr);
    ASSERT_EQ(translated, expected_paddr);

    // 测试页内偏移
    uint32_t offset = 0x555;
    translated = paging_manager->translate_virtual_to_physical(vaddr + offset);
    ASSERT_EQ(translated, expected_paddr + offset);
  }
}

TEST_F(PageTableTest, PagePermissions) {
  // Test case 4: 页面权限测试
  uint32_t vaddr = 0x00004000;
  uint32_t pframe = 0x400;

  // 测试只读页面映射
  ASSERT_TRUE(
      paging_manager->map_page(vaddr, pframe, false, true)); // writable = false

  // 测试用户访问权限页面映射
  ASSERT_TRUE(paging_manager->map_page(vaddr + PAGE_SIZE, pframe + 1, true,
                                       true)); // user_accessible = true

  // 测试内核页面映射
  ASSERT_TRUE(paging_manager->map_page(vaddr + 2 * PAGE_SIZE, pframe + 2, true,
                                       false)); // user_accessible = false
}

TEST_F(PageTableTest, UnmapPages) {
  uint32_t vaddr = 0x00005000;
  uint32_t pframe = 0x500;

  // 首先创建映射
  ASSERT_TRUE(paging_manager->map_page(vaddr, pframe));
  ASSERT_EQ(paging_manager->translate_virtual_to_physical(vaddr),
            pframe * PAGE_SIZE);

  // 取消映射
  paging_manager->unmap_page(vaddr);

  // 验证页面已被取消映射
  ASSERT_EQ(paging_manager->translate_virtual_to_physical(vaddr), ~0U);
}