/**
 * @file test_polarity.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-07-23
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "base_type_convert.hpp"
#include "mexception.hpp"
#include "type.hpp"
#include <iostream>
#include <regex>
#include <string>

using namespace rule_applicator;
using namespace utils;

struct DemoComponentInfo {
  std::string position;
  int ocrLineCount;
  int pads_count;
  std::pair<int, int> pad_groups_pins_count;
};

static bool IsConditionMet(const DemoComponentInfo &info,
                           const Condition &cond) {
  switch (cond.field) {
  case PolarityFieldType::Position: {
    switch (cond.op) {
    case RuleOperator::RegexMatch:
      return std::regex_match(info.position, std::regex(cond.value));
    case RuleOperator::RegexNotMatch:
      return !std::regex_match(info.position, std::regex(cond.value));
    default:
      throw exception::InvalidValueException("Unsupported op type!");
    }
  }
  case PolarityFieldType::OCRLineCount: {
    auto vvalue = utils::convert_string_to_number(cond.value);
    auto value = exception::get_or_throw<int>(vvalue);
    switch (cond.op) {
    case RuleOperator::Equals:
      return info.ocrLineCount == value;
    case RuleOperator::NotEquals:
      return info.ocrLineCount != value;
    case RuleOperator::GreaterThan:
      return info.ocrLineCount > value;
    case RuleOperator::LessThan:
      return info.ocrLineCount < value;
    case RuleOperator::GreaterThanOrEqual:
      return info.ocrLineCount >= value;
    case RuleOperator::LessThanOrEqual:
      return info.ocrLineCount <= value;
    default:
      throw exception::InvalidValueException("Unsupported op type!");
    }
  }
  case PolarityFieldType::PadParity: {
    switch (cond.op) {
    case RuleOperator::IsOdd:
      return info.pads_count % 2 == 1;
    case RuleOperator::IsEven:
      return info.pads_count % 2 == 0;
    default:
      throw exception::InvalidValueException("Unsupported op type!");
    }
  }
  }
  return false;
}

static inline bool IsRuleMatched(const DemoComponentInfo &info,
                                 const Rule &rule) {
  // conditions之间是与逻辑
  for (const Condition &cond : rule.conditions) {
    if (!IsConditionMet(info, cond)) {
      return false;
    }
  }
  return true;
}

static inline Polarity DeterminePolarity(const DemoComponentInfo &info,
                                         const std::vector<Rule> &rules) {
  // rules之间是或逻辑
  for (const auto &rule : rules) {
    if (IsRuleMatched(info, rule)) {
      return rule.polarity;
    }
  }
  return Polarity::UNKNOWN;
}

int main() {
  // demo datas
  const std::vector<DemoComponentInfo> components = {
      {"R12", 3}, {"C1", 0}, {"RF2", 2}, {"RD3", 2}, {"U1", 1}};

  // demo rules（所有规则只要有一条满足就会返回对应规则的 Polarity type）
  std::vector<Rule> rules;

  // Rule 1: position 以 R 开头且不是 RF 开头的无极性
  rules.push_back(Rule{
      {Condition{PolarityFieldType::Position, RuleOperator::RegexMatch, "^R.*"},
       Condition{PolarityFieldType::Position, RuleOperator::RegexNotMatch,
                 "^RF.*"}},
      Polarity::ABSENT});

  // Rule 2: position 以 C 开头的或 MC 开头的且无多行文字则无极性
  rules.push_back(Rule{{Condition{PolarityFieldType::Position,
                                  RuleOperator::RegexMatch, "^(C|MC).*"}},
                       Polarity::ABSENT});

  // Rule 3: ocrLineCount 只有一行
  rules.push_back(Rule{
      {Condition{PolarityFieldType::OCRLineCount, RuleOperator::Equals, "1"}},
      Polarity::ABSENT});

  // Rule 4: pad 数量为奇数
  rules.push_back(
      Rule{{Condition{PolarityFieldType::PadParity, RuleOperator::IsOdd, ""}},
           Polarity::ABSENT});

  for (const auto &info : components) {
    Polarity polarity = DeterminePolarity(info, rules);
    std::cout << static_cast<int>(polarity) << std::endl;
  }
  return 0;
}
