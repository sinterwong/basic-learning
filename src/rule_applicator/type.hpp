/**
 * @file type.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-07-23
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __SRC_RULE_APPLICATOR_TYPE_HPP_
#define __SRC_RULE_APPLICATOR_TYPE_HPP_

#include <any>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
namespace rule_applicator {
using DataField = std::any;
using DataMap = std::unordered_map<std::string, DataField>;

enum class Polarity : uint8_t { NONE = 0, ABSENT, PRESENT, UNKNOWN };

struct Condition {
  std::string field;
  std::string op;
  std::string value;
};
struct Rule {
  std::vector<Condition> conditions;
  Polarity polarity;
};

} // namespace rule_applicator

#endif