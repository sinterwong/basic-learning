#include "library_manager.hpp"
#include <gtest/gtest.h>

using namespace template_mp::specialization;

TEST(TestLibraryManager, TestLibraryManager) {
  auto studentRegular =
      GetBorrowPolicy<BorrowerType::STUDENT, BookType::REGULAR>();

  auto teacherRegular =
      GetBorrowPolicy<BorrowerType::TEACHER, BookType::REGULAR>();

  auto visitorRegular =
      GetBorrowPolicy<BorrowerType::VISITOR, BookType::REGULAR>();

  auto studentReference =
      GetBorrowPolicy<BorrowerType::STUDENT, BookType::REFERENCE>();

  ASSERT_EQ(studentRegular.duration, 14);
  ASSERT_TRUE(studentRegular.renewable);
  ASSERT_EQ(teacherRegular.duration, 28);
  ASSERT_TRUE(teacherRegular.renewable);
  ASSERT_EQ(visitorRegular.duration, 7);
  ASSERT_FALSE(visitorRegular.renewable);
  ASSERT_EQ(studentReference.duration, 2);
  ASSERT_FALSE(studentReference.renewable);
}
