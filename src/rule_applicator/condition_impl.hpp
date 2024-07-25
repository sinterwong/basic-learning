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

class Equals : public IConditionEvaluator {
public:
  bool evaluate(const DataField &data, const std::string &value) const override;
};

class NotEquals : public IConditionEvaluator {
public:
  bool evaluate(const DataField &data, const std::string &value) const override;
};

class GreaterThan : public IConditionEvaluator {
public:
  bool evaluate(const DataField &data, const std::string &value) const override;
};

class LessThan : public IConditionEvaluator {
public:
  bool evaluate(const DataField &data, const std::string &value) const override;
};

class GreaterThanOrEqual : public IConditionEvaluator {
public:
  bool evaluate(const DataField &data, const std::string &value) const override;
};

class LessThanOrEqual : public IConditionEvaluator {
public:
  bool evaluate(const DataField &data, const std::string &value) const override;
};

class IsOdd : public IConditionEvaluator {
public:
  bool evaluate(const DataField &data, const std::string &value) const override;
};

class IsEven : public IConditionEvaluator {
public:
  bool evaluate(const DataField &data, const std::string &value) const override;
};

} // namespace rule_applicator

#endif