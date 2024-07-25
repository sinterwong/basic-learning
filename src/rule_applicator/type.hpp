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

enum class Polarity : uint8_t { NONE = 0, ABSENT, PRESENT, UNKNOWN };

enum class PolarityFieldType : uint8_t {
  Position = 0,
  OCRLineCount,
  PadsCount,
  PadParity
};

struct Condition {
  PolarityFieldType field;
  std::string op;
  std::string value;
};

struct Rule {
  std::vector<Condition> conditions;
  Polarity polarity;
};

struct DemoComponentInfo {
  std::string position;
  int ocrLineCount;
  int pads_count;
};

} // namespace rule_applicator

#endif