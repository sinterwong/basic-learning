/**
 * @file memory_model.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-26
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __OP_SYSTEM_MEMORY_MODEL_HPP_
#define __OP_SYSTEM_MEMORY_MODEL_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

namespace op_system::mem {

// memory block size
constexpr size_t BLOCK_SIZE = 4096;

// memory page
struct MemoryBlock {
  std::vector<uint8_t> data;

  // init to zero
  MemoryBlock() : data(BLOCK_SIZE, 0x0) {}
};

class MyMemoryModel {
private:
  // memory page table
  std::unordered_map<uint64_t, std::shared_ptr<MemoryBlock>> memory;

public:
  /**
   * @brief 读取内存中的数据到pData
   *
   * @param pData
   * @param addr
   * @param len
   */
  void read(uint8_t *pData, uint64_t addr, uint32_t len) {
    while (len > 0) {
      // 获取内存块的索引
      uint64_t blockIndex = addr / BLOCK_SIZE;

      // 获取内存块中实际地址的偏移
      uint64_t offset = addr % BLOCK_SIZE;

      // 本次要读取的长度（这里考虑了跨内存块的情况）
      uint64_t readLength =
          std::min(static_cast<uint64_t>(len), BLOCK_SIZE - offset);

      // 根据内存索引获取内存页并根据偏移定位到目标起始地址
      auto it = memory.find(blockIndex);
      if (it != memory.end()) {
        auto block = it->second;
        std::memcpy(pData, &block->data[offset], readLength);
      } else {
        // 对于未初始化的区域默认读出来0
        std::memset(pData, 0x0, readLength);
      }

      // 更新相关指针和剩余数据长度
      addr += readLength;
      pData += readLength;
      len -= readLength;
    }
  }

  /**
   * @brief 写入数据到内存中
   *
   * @param pData
   * @param addr
   * @param len
   */
  void write(const uint8_t *pData, uint64_t addr, uint32_t len) {
    while (len > 0) {
      uint64_t blockIndex = addr / BLOCK_SIZE;
      uint64_t offset = addr % BLOCK_SIZE;
      uint64_t writeLength =
          std::min(static_cast<uint64_t>(len), BLOCK_SIZE - offset);

      auto &block = memory[blockIndex];
      if (!block) {
        block = std::make_shared<MemoryBlock>();
      }

      std::memcpy(&block->data[offset], pData, writeLength);

      addr += writeLength;
      pData += writeLength;
      len -= writeLength;
    }
  }
};
} // namespace op_system::mem
#endif