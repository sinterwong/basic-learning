#ifndef __SRC_RULE_APPLICATOR_CONDITION_HPP_
#define __SRC_RULE_APPLICATOR_CONDITION_HPP_

#include "type.hpp"

namespace rule_applicator {
class IConditionEvaluator {
public:
  virtual ~IConditionEvaluator() = default;
  virtual bool evaluate(const DataField &data,
                        const std::string &value) const = 0;
};

class RuleApplicatorRegistrar {
public:
  static RuleApplicatorRegistrar &getInstance() {
    static RuleApplicatorRegistrar instance;
    return instance;
  }

  RuleApplicatorRegistrar(const RuleApplicatorRegistrar &) = delete;
  RuleApplicatorRegistrar &operator=(const RuleApplicatorRegistrar &) = delete;

private:
  RuleApplicatorRegistrar();
};

} // namespace rule_applicator

#endif