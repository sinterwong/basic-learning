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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <ostream>
#include <unordered_map>
#include <vector>

namespace algo_and_ds::practices {

// 内存块的大小
constexpr size_t BLOCK_SIZE = 4096;

// 定义内存页结构
struct MemoryBlock {
  std::vector<uint8_t> data;

  // 初始化内存为0
  MemoryBlock() : data(BLOCK_SIZE, 0x0) {}
};

// 内存模型
class MyMemoryModel {
private:
  // 简易的内存页表
  std::unordered_map<uint64_t, MemoryBlock> memory;

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
      MemoryBlock &block = memory.at(blockIndex);
      std::memcpy(pData, &block.data[offset], readLength);

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

      MemoryBlock &block = memory[blockIndex];
      std::memcpy(&block.data[offset], pData, writeLength);

      addr += writeLength;
      pData += writeLength;
      len -= writeLength;
    }
  }
};
} // namespace algo_and_ds::practices

int main() {
  using algo_and_ds::practices::MyMemoryModel;
  MyMemoryModel memoryModel;

  // 测试数据和地址
  const uint64_t testAddr = 0x12345678;
  const uint32_t testDataLen = 5;
  uint8_t testData[testDataLen] = {1, 2, 3, 4, 5};

  // 写入测试数据到内存模型
  memoryModel.write(testData, testAddr, testDataLen);

  // 从内存模型读取数据
  uint8_t readData[testDataLen] = {0}; // 用于存储读取数据的缓冲区
  memoryModel.read(readData, testAddr, testDataLen);

  // 比较读取的数据与原始数据是否相同
  if (std::memcmp(testData, readData, testDataLen) == 0) {
    std::cout << "Test passed!" << std::endl;
  } else {
    std::cout << "Test failed!" << std::endl;
  }
  return 0;
}