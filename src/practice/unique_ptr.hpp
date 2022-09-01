/**
 * @file unique_ptr.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-09-01
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __PRACTICE_UNIQUE_PTR_HPP_
#define __PRACTICE_UNIQUE_PTR_HPP_

#include <iostream>

namespace practice {

template <typename Tp>
class unique_ptr {
private:
  Tp* pointer;

};

}

#endif