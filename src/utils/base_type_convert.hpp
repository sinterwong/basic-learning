/**
 * @file base_type_convert.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-07-24
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __UTILS_BASE_TYPE_CONVERT_H_
#define __UTILS_BASE_TYPE_CONVERT_H_

#include <regex>
#include <stdexcept>
#include <string>
#include <variant>

namespace utils {

inline std::variant<int, float, std::string>
convert_string_to_number(const std::string &s) {
  // regular expression for matching numbers (including scientific notation)
  const std::regex number_regex(
      R"(^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$)");

  // not a number
  if (!std::regex_match(s, number_regex)) {
    return s;
  }

  try {
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find('E') == std::string::npos) {
      return std::stoi(s);
    } else {
      return std::stof(s);
    }
  } catch (const std::invalid_argument &e) {
    return s; // conversion failed, return original string
  } catch (const std::out_of_range &e) {
    return s; // number out of range, return original string
  }
}
} // namespace utils

#endif