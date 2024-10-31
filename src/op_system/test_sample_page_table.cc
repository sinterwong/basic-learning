#include "sample_page_table.hpp"
#include <gtest/gtest.h>
#include <stdexcept>

using namespace op_system;

class SamplePageTableTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(SamplePageTableTest, OneStage) {
  virtual_mem::SamplePageTable page_table;

  // 0x0000F000(vaddr pfn) -> 0x00100(paddr pfn)
  virtual_mem::SamplePageTableEntry entry;
  entry.pfn = 0x00100;
  entry.present = true;
  entry.writable = true;
  entry.executable = false;

  uint32_t vpn = 0x0F000 >> 12;

  // 0x0000F 000 -> 0x00100 000
  page_table.set_entry(vpn, entry);

  // 0x0000F123(vir addr) -> 0x00100123(phy addr)
  uint32_t virtual_address = 0x0000F123;
  uint32_t physical_address = page_table.get_physical_address(virtual_address);
  ASSERT_EQ(physical_address, 0x00100123);

  // page fault
  virtual_address = 0x00001000;
  ASSERT_THROW(page_table.get_physical_address(virtual_address),
               std::runtime_error);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
