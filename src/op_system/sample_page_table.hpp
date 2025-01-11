#ifndef __OP_SYSTEM_PAGE_TABLE_HPP_
#define __OP_SYSTEM_PAGE_TABLE_HPP_

#include <cstdint>
#include <stdexcept>
#include <vector>
namespace op_system::virtual_mem {
/**
 * @brief 知识描述：虚拟内存与页表
  1.现代操作系统使用虚拟内存来管理进程的地址空间，使得每个进程认为自己独占整个内存空间。这种机制需要将虚拟地址转换为物理地址，而页表正是实现这种转换的核心数据结构。

  2.假设不使用页表，而是采用简单的线性映射（一一映射），将会消耗巨大的内存资源。

  例如，在32位系统中，虚拟地址空间大小为2^32字节。如果每个虚拟地址都直接映射到物理地址，需
  要2^32个地址项，每个地址项至少4字节（32位物理地址），总共需要4*2^32字节的内存，这显然是不切实际的。

  3.页表提供了一种分而治之的机制，将虚拟地址空间和物理地址空间都划分为固定大小的页面。每个进程拥有自己的页表，用于管理其虚拟地址空间到物理地址空间的映射关系。

  4.虚拟地址被划分为页号（Page Number）和页内偏移（Page
 Offset）。页大小通常是4KB (2^12字节) 或更大。

  例如，在32位系统中，如果页大小为4KB，那么虚拟地址的高20位表示页号，低12位表示页内偏移。
     [例：0xABCDEFFF，其中0xABCDE是页号，0xFFF是页内偏移]。

  5.页表项（Page Table
 Entry，PTE）：页表是由许多PTE组成的。每个PTE对应一个虚拟页面。
     PTE中存储了物理页框号（Page Frame Number，PFN）以及一些标志位，例如：
       - Present/Absent位：表示该虚拟页面是否当前在物理内存中。
       - Dirty位：表示该页面是否被修改过。
       - Accessed位：表示该页面是否被访问过。
       - Protection位：表示该页面的读/写/执行权限。
     最终的物理地址是通过将PFN左移页大小的位数，然后加上页内偏移量得到的。

  6.多级页表：为了避免单级页表占用过大的内存空间，现代操作系统通常采用多级页表（例如二级页表、三级页表）。
     多级页表可以按需加载，只加载当前需要的页表项，从而减少内存消耗。

 7.TLB（Translation Lookaside
 Buffer，转换后备缓冲器）：为了加速地址转换过程，CPU内部集成了一个叫做TLB的缓存。TLB存储了最近使用的虚拟地址到物理地址的映射关系。如果TLB命中，可以直接获取物理地址，无需访问页表。

  8.
 页面置换算法：当物理内存不足时，操作系统需要将一些页面从物理内存换出到磁盘，以便为新的页面腾出空间。
     这涉及到页面置换算法，例如FIFO（先进先出）、LRU（最近最少使用）等。
 */

// 假设地址空间为32位
struct SamplePageTableEntry {
  uint32_t pfn;
  bool present;
  bool dirty;
  bool accessed;
  bool writable;
  bool executable;
};

class SamplePageTable {
private:
  // page table entry
  std::vector<SamplePageTableEntry> page_table;

  // page table size
  uint32_t page_table_size;

public:
  SamplePageTable() : page_table(1 << 20), page_table_size(1 << 20) {}

  SamplePageTableEntry &get_entry(uint32_t vpn) {

    if (vpn >= page_table_size) {
      throw std::out_of_range("vpn out of range");
    }
    return page_table.at(vpn);
  }

  void set_entry(uint32_t vpn, SamplePageTableEntry entry) {
    if (vpn >= page_table_size) {
      throw std::out_of_range("vpn out of range");
    }
    page_table.at(vpn) = entry;
  }

  // get physical address
  uint32_t get_physical_address(uint32_t virtual_address) {
    // get page index
    uint32_t vpn = virtual_address >> 12;
    // get offset
    uint32_t offset = virtual_address & 0xFFF;
    // get entry
    SamplePageTableEntry entry = get_entry(vpn);

    if (!entry.present) {
      throw std::runtime_error("Page fault");
    }
    uint32_t physical_address = (entry.pfn << 12) + offset;
    return physical_address;
  }

  bool is_valid_address(uint32_t virtual_address) {
    uint32_t vpn = virtual_address >> 12;
    SamplePageTableEntry entry = get_entry(vpn);
    return entry.present;
  }
};

} // namespace op_system::virtual_mem

#endif