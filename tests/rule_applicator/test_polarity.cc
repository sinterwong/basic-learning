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

#include "oop/factory.hpp"
#include "rule_applicator/condition.hpp"
#include "rule_applicator/type.hpp"
#include "utils/mexception.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace rule_applicator;

using DataMap = std::unordered_map<PolarityFieldType, DataField>;

bool IsConditionMet(const DataMap &data, const Condition &cond) {
  auto it = data.find(cond.field);
  if (it == data.end()) {
    return false;
  }

  auto evaluator =
      oop::factory::ObjFactory::createObj<IConditionEvaluator>(cond.op);
  if (!evaluator) {
    throw utils::exception::NullPointerException(
        "evaluator is a null pointer!");
  }
  return evaluator->evaluate(it->second, cond.value);
}

static inline bool IsRuleMatched(const DataMap &info, const Rule &rule) {
  // conditions之间是与逻辑
  for (const Condition &cond : rule.conditions) {
    if (!IsConditionMet(info, cond)) {
      return false;
    }
  }
  return true;
}

static inline Polarity DeterminePolarity(const DataMap &info,
                                         const std::vector<Rule> &rules) {
  // rules之间是或逻辑
  for (const auto &rule : rules) {
    if (IsRuleMatched(info, rule)) {
      return rule.polarity;
    }
  }
  return Polarity::UNKNOWN;
}

TEST(PolarityTest, Normal) {
  RuleApplicatorRegistrar::getInstance();

  // demo datas
  const std::vector<DemoComponentInfo> components = {
      {"R12", 3, 4}, {"C1", 0, 5}, {"RF2", 2, 19}, {"RD3", 2, 8}, {"U1", 2, 3}};

  std::vector<DataMap> componentInfos;
  for (const auto &component : components) {
    DataMap componentInfo;
    componentInfo[PolarityFieldType::Position] = component.position;
    componentInfo[PolarityFieldType::OCRLineCount] = component.ocrLineCount;
    componentInfo[PolarityFieldType::PadsCount] = component.pads_count;
    componentInfos.push_back(componentInfo);
  }

  // demo rules（所有规则只要有一条满足就会返回对应规则的 Polarity type）
  std::vector<Rule> rules;

  // Rule 1: position 以 R 开头且不是 RF 开头的无极性
  rules.push_back(
      Rule{{Condition{PolarityFieldType::Position, "RegexMatch", "^R.*"},
            Condition{PolarityFieldType::Position, "RegexNotMatch", "^RF.*"}},
           Polarity::ABSENT});

  // Rule 2: position 以 C 开头的或 MC 开头的且无多行文字则无极性
  rules.push_back(
      Rule{{Condition{PolarityFieldType::Position, "RegexMatch", "^(C|MC).*"}},
           Polarity::ABSENT});

  // Rule 3: ocrLineCount 只有一行	  for (const auto &info :
  // componentInfos) {
  rules.push_back(
      Rule{{Condition{PolarityFieldType::OCRLineCount, "Equals", "1"}},
           Polarity::ABSENT});

  // Rule 4: pad 数量为奇数
  rules.push_back(Rule{{Condition{PolarityFieldType::PadParity, "IsOdd", ""}},
                       Polarity::ABSENT});

  for (const auto &info : componentInfos) {
    Polarity polarity = DeterminePolarity(info, rules);
    std::cout << static_cast<int>(polarity) << std::endl;
  }
}
