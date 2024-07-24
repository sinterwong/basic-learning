#include "condition_impl.hpp"
#include "oop/factory.hpp"
#include <regex>

using namespace oop::factory;

namespace rule_applicator {
bool RegexMatch::evaluate(DataField const &data,
                          std::string const &value) const {
  if (data.type() == typeid(std::string)) {
    const std::string &str = std::any_cast<const std::string &>(data);
    return std::regex_match(str, std::regex(value));
  }
  return false;
}

bool RegexNotMatch::evaluate(DataField const &data,
                             std::string const &value) const {
  if (data.type() == typeid(std::string)) {
    const std::string &str = std::any_cast<const std::string &>(data);
    return !std::regex_match(str, std::regex(value));
  }
  return false;
}

RuleApplicatorRegistrar::RuleApplicatorRegistrar() {
  BasicLearningModuleRegister(RegexMatch);
  BasicLearningModuleRegister(RegexNotMatch);
}
} // namespace rule_applicator
