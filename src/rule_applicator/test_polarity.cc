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

#include "condition.hpp"
#include "oop/factory.hpp"
#include "type.hpp"
#include "utils/mexception.hpp"
#include <iostream>
#include <string>
#include <utility>

using namespace rule_applicator;

struct DemoComponentInfo {
  std::string position;
  int ocrLineCount;
  int pads_count;
  std::pair<int, int> pad_groups_pins_count;
};

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

int main() {
  RuleApplicatorRegistrar::getInstance();

  // demo datas
  const std::vector<DemoComponentInfo> components = {
      {"R12", 3, 4, std::make_pair(3, 4)},
      {"C1", 0, 5, std::make_pair(4, 4)},
      {"RF2", 2, 19, std::make_pair(2, 4)},
      {"RD3", 2, 8, std::make_pair(3, 3)},
      {"U1", 1, 3, std::make_pair(3, 5)}};

  std::vector<DataMap> componentInfos;
  for (const auto &component : components) {
    DataMap componentInfo;
    componentInfo["Position"] = component.position;
    componentInfo["OcrLineCount"] = component.ocrLineCount;
    componentInfo["PadsCount"] = component.pads_count;
    componentInfo["PinsPair"] = component.pad_groups_pins_count;
    componentInfos.push_back(componentInfo);
  }

  // demo rules（所有规则只要有一条满足就会返回对应规则的 Polarity type）
  std::vector<Rule> rules;

  // Rule 1: position 以 R 开头且不是 RF 开头的无极性
  rules.push_back(Rule{{Condition{"Position", "RegexMatch", "^R.*"},
                        Condition{"Position", "RegexNotMatch", "^RF.*"}},
                       Polarity::ABSENT});

  for (const auto &info : componentInfos) {
    Polarity polarity = DeterminePolarity(info, rules);
    std::cout << static_cast<int>(polarity) << std::endl;
  }

  return 0;
}
