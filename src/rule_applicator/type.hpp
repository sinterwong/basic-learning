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

#include <cstdint>
#include <string>
#include <vector>
namespace rule_applicator {
enum class RuleOperator {
  Equals,             // ==
  NotEquals,          // !=
  GreaterThan,        // >
  LessThan,           // <
  GreaterThanOrEqual, // >=
  LessThanOrEqual,    // <=
  RegexMatch,         // regex match
  RegexNotMatch,      // regex not match
  IsOdd,              // odd
  IsEven              // even
};

enum class Polarity : uint8_t { NONE = 0, ABSENT, PRESENT, UNKNOWN };

enum class PolarityFieldType : uint8_t {
  Position = 0,
  OCRLineCount,
  PadParity
};

struct Condition {
  PolarityFieldType field;
  RuleOperator op;
  std::string value;
};

struct Rule {
  std::vector<Condition> conditions;
  Polarity polarity;
};

} // namespace rule_applicator

#endif