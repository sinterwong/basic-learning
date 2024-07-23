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

#include "mexception.hpp"
#include "type.hpp"
#include <iostream>
#include <regex>

using namespace rule_applicator;
using namespace utils::exception;

static bool IsConditionMet(const DemoComponentInfo &info,
                           const Condition &cond) {
  switch (cond.field) {
  case PolarityFieldType::Position: {
    const auto &value = get_or_throw<std::string>(cond.value);
    switch (cond.op) {
    case Operator::RegexMatch:
      return std::regex_match(info.position, std::regex(value));
    case Operator::RegexNotMatch:
      return !std::regex_match(info.position, std::regex(value));
    default:
      throw InvalidValueException("Unsupported op type!");
    }
  }
  case PolarityFieldType::OCRLineCount: {
    const auto &value = get_or_throw<int>(cond.value);
    switch (cond.op) {
    case Operator::Equals:
      return info.ocrLineCount == value;
    case Operator::NotEquals:
      return info.ocrLineCount != value;
    case Operator::GreaterThan:
      return info.ocrLineCount > value;
    case Operator::LessThan:
      return info.ocrLineCount < value;
    case Operator::GreaterThanOrEqual:
      return info.ocrLineCount >= value;
    case Operator::LessThanOrEqual:
      return info.ocrLineCount <= value;
    default:
      throw InvalidValueException("Unsupported op type!");
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
      {Condition{PolarityFieldType::Position, Operator::RegexMatch, "^R.*"},
       Condition{PolarityFieldType::Position, Operator::RegexNotMatch,
                 "^RF.*"}},
      Polarity::ABSENT});

  // Rule 2: position 以 C 开头的或 MC 开头的且无多行文字则无极性
  rules.push_back(Rule{{Condition{PolarityFieldType::Position,
                                  Operator::RegexMatch, "^(C|MC).*"}},
                       Polarity::ABSENT});

  // Rule 3: ocrLineCount 只有一行
  rules.push_back(
      Rule{{Condition{PolarityFieldType::OCRLineCount, Operator::Equals, 1}},
           Polarity::ABSENT});

  for (const auto &info : components) {
    Polarity polarity = DeterminePolarity(info, rules);
    std::cout << static_cast<int>(polarity) << std::endl;
  }
  return 0;
}
