/**
 * @file library_manager.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-08-22
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "library_manager.hpp"

namespace template_mp::specialization {

// student with regular
template <>
BorrowPolicy GetBorrowPolicy<BorrowerType::STUDENT, BookType::REGULAR>() {
  return BorrowPolicy{14, true}; // 14天，可续借
}

// teacher with regular
template <>
BorrowPolicy GetBorrowPolicy<BorrowerType::TEACHER, BookType::REGULAR>() {
  return BorrowPolicy{28, true}; // 28天，可续借
}

// vistor with regular
template <>
BorrowPolicy GetBorrowPolicy<BorrowerType::VISITOR, BookType::REGULAR>() {
  return BorrowPolicy{7, false}; // 7天，不可续借
}

// student with reference
template <>
BorrowPolicy GetBorrowPolicy<BorrowerType::STUDENT, BookType::REFERENCE>() {
  return BorrowPolicy{2, false}; // 2天，不可续借
}
} // namespace template_mp::specialization