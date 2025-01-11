#ifndef __OP_SYSTEM_TWO_STAGE_PAGE_TABLE_HPP_
#define __OP_SYSTEM_TWO_STAGE_PAGE_TABLE_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>
namespace op_system::virtual_mem {
/**
 * @brief 知识描述：二级页表
 * 一级页表把所有虚拟地址空间的映射都放在一个巨大的页表中。每个虚拟页面都需要一个页表项，即使该页面未被使用。假设你的系统有4GB的虚拟地址空间，页面大小为4KB，那么你需要
 * 1M(4GB/4KB)个页表项。如果每个页表项是4字节，那么这个页表就需要4MB的连续物理内存。
 *
 * 二级页表把这个巨大的页表拆分成多个更小的页表。页目录充当一级索引，每个页目录项指向一个页表。只有当页目录项对应的页表被用到时，这个页表才需要分配物理内存。
 *
 * 例如只使用了16MB的虚拟地址空间，那么只需要4个页表(16MB/4MB)。在一级页表中，即使只使用了
 * 16MB，仍然需要4MB大小的页表来表达所以可能的情况。而在二级页表中，只需要4个页表*4KB=16KB的内存，加上
 * 页目录4KB 的内存，共 20KB。 相比一级页表的 4MB，节省了大量内存
 *
 * 所以，二级页表的优势在于：不需要一次性分配所有可能的页表，只有实际使用的虚拟地址空间对应的页表才需要分配，从而节省了物理内存。未使用的虚拟地址空间对应的页表不需要存在，相应的页目录项的
 * present 位设置为 0。
 */

// 页大小和基本常量定义
constexpr size_t PAGE_SIZE = 4096;
constexpr size_t ENTRIES_PER_TABLE = 1024; // 2^10
constexpr size_t KERNEL_RESERVED_PAGES = 0x100;

// 页表项结构体
struct PageTableEntry {
  uint32_t frame_number : 20;   // 物理页框号
  uint32_t present : 1;         // 存在位
  uint32_t writable : 1;        // 可写位
  uint32_t user_accessible : 1; // 用户态可访问位
  uint32_t write_through : 1;   // 写透缓存位
  uint32_t cache_disabled : 1;  // 缓存禁用位
  uint32_t accessed : 1;        // 访问位
  uint32_t dirty : 1;           // 修改位
  uint32_t global : 1;          // 全局位
  uint32_t available : 3;       // 可用位
};

// 页目录项结构体
struct PageDirectoryEntry {
  uint32_t pt_frame_number : 20; // 页表的物理页框号
  uint32_t present : 1;
  uint32_t writable : 1;
  uint32_t user_accessible : 1;
  uint32_t write_through : 1;
  uint32_t cache_disabled : 1;
  uint32_t accessed : 1;
  uint32_t available : 6;
};

// 页表类
class PageTable {
public:
  PageTableEntry &operator[](size_t index) { return entries[index]; }

  const PageTableEntry &operator[](size_t index) const {
    return entries[index];
  }

private:
  PageTableEntry entries[ENTRIES_PER_TABLE];
};

// 页面管理器类
class PagingManager {
public:
  explicit PagingManager(size_t total_pages)
      : total_physical_pages(total_pages),
        physical_page_bitmap(total_pages, false) {
    // 预留内核页
    for (size_t i = 0; i < KERNEL_RESERVED_PAGES; ++i) {
      physical_page_bitmap[i] = true;
    }

    // 分配页目录使用的物理页
    page_directory_frame = allocate_physical_frame();
    if (page_directory_frame == INVALID_FRAME) {
      throw std::runtime_error("Failed to allocate page directory");
    }
  }

  uint32_t translate_virtual_to_physical(uint32_t virtual_address) const {
    auto [pd_index, pt_index, offset] =
        decompose_virtual_address(virtual_address);

    // 检查页目录项
    const auto &pde = page_directory[pd_index];
    if (!pde.present) {
      return INVALID_ADDRESS;
    }

    // 获取对应的页表
    const PageTable *pt = get_page_table(pde.pt_frame_number);
    const auto &pte = (*pt)[pt_index];
    if (!pte.present) {
      return INVALID_ADDRESS;
    }

    return (pte.frame_number << 12) | offset;
  }

  bool map_page(uint32_t virtual_address, uint32_t physical_frame,
                bool writable = true, bool user_accessible = true) {
    auto [pd_index, pt_index, offset] =
        decompose_virtual_address(virtual_address);

    // 确保物理页框有效
    if (physical_frame >= total_physical_pages) {
      return false;
    }

    // 检查并创建页表
    if (!page_directory[pd_index].present) {
      uint32_t new_pt_frame = allocate_physical_frame();
      if (new_pt_frame == INVALID_FRAME) {
        return false;
      }

      // 初始化新页表
      PageTable *new_pt = get_page_table(new_pt_frame);
      std::memset(new_pt, 0, sizeof(PageTable));

      // 设置页目录项
      auto &pde = page_directory[pd_index];
      pde.pt_frame_number = new_pt_frame;
      pde.present = 1;
      pde.writable = writable;
      pde.user_accessible = user_accessible;
    }

    // 设置页表项
    PageTable *pt = get_page_table(page_directory[pd_index].pt_frame_number);
    auto &pte = (*pt)[pt_index];
    pte.frame_number = physical_frame;
    pte.present = 1;
    pte.writable = writable;
    pte.user_accessible = user_accessible;

    return true;
  }

  void unmap_page(uint32_t virtual_address) {
    auto [pd_index, pt_index, offset] =
        decompose_virtual_address(virtual_address);

    if (!page_directory[pd_index].present) {
      return;
    }

    PageTable *pt = get_page_table(page_directory[pd_index].pt_frame_number);
    (*pt)[pt_index].present = 0;
  }

private:
  static constexpr uint32_t INVALID_FRAME = ~0U;
  static constexpr uint32_t INVALID_ADDRESS = ~0U;

  struct VirtualAddressComponents {
    uint32_t pd_index;
    uint32_t pt_index;
    uint32_t offset;
  };

  VirtualAddressComponents decompose_virtual_address(uint32_t addr) const {
    return {
        (addr >> 22) & 0x3FF, // pd_index
        (addr >> 12) & 0x3FF, // pt_index
        addr & 0xFFF          // offset
    };
  }

  uint32_t allocate_physical_frame() {
    for (size_t i = KERNEL_RESERVED_PAGES; i < total_physical_pages; ++i) {
      if (!physical_page_bitmap[i]) {
        physical_page_bitmap[i] = true;
        return i * PAGE_SIZE;
      }
    }
    return INVALID_FRAME;
  }

  void free_physical_frame(uint32_t frame) {
    if (frame < total_physical_pages) {
      physical_page_bitmap[frame] = false;
    }
  }

  PageTable *get_page_table(uint32_t frame_number) const {
    // 实际系统中这里应该返回物理地址映射后的虚拟地址
    // 这里假设页表被映射到某个固定的虚拟地址范围
    static std::unordered_map<uint32_t, std::unique_ptr<PageTable>> page_tables;

    auto it = page_tables.find(frame_number);
    if (it == page_tables.end()) {
      auto [inserted_it, _] =
          page_tables.emplace(frame_number, std::make_unique<PageTable>());
      return inserted_it->second.get();
    }
    return it->second.get();
  }

  const size_t total_physical_pages;
  std::vector<bool> physical_page_bitmap; // 管理物理页面分配
  uint32_t page_directory_frame;
  PageDirectoryEntry page_directory[ENTRIES_PER_TABLE];
};

} // namespace op_system::virtual_mem

#endif