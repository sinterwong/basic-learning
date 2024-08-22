#include "library_manager.hpp"
#include <iostream>

using namespace template_mp::specialization;

int main() {
  auto studentRegular =
      GetBorrowPolicy<BorrowerType::STUDENT, BookType::REGULAR>();

  auto teacherRegular =
      GetBorrowPolicy<BorrowerType::TEACHER, BookType::REGULAR>();

  std::cout << "Student borrowing regular book: " << studentRegular.duration
            << " days, renewable: " << studentRegular.renewable << std::endl;

  std::cout << "Teacher borrowing regular book: " << teacherRegular.duration
            << " days, renewable: " << teacherRegular.renewable << std::endl;

  return 0;
}