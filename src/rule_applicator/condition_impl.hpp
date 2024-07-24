#ifndef __SRC_RULE_APPLICATOR_REGEX_MATCH_CONDITION_HPP_
#define __SRC_RULE_APPLICATOR_REGEX_MATCH_CONDITION_HPP_

#include "condition.hpp"

namespace rule_applicator {
class RegexMatch : public IConditionEvaluator {
public:
  bool evaluate(const DataField &data, const std::string &value) const override;
};

class RegexNotMatch : public IConditionEvaluator {
public:
  bool evaluate(const DataField &data, const std::string &value) const override;
};

} // namespace rule_applicator

#endif