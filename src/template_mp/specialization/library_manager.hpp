/**
 * @file library_manager.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-08-22
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef _TEMPLATE_MP_SPECIALIZATION_HPP_
#define _TEMPLATE_MP_SPECIALIZATION_HPP_

namespace template_mp::specialization {
enum class BorrowerType { STUDENT, TEACHER, VISITOR };
enum class BookType { REGULAR, REFERENCE, RARE };

struct BorrowPolicy {
  int duration;
  bool renewable;
};

template <BorrowerType, BookType> BorrowPolicy GetBorrowPolicy();

} // namespace template_mp::specialization
#endif