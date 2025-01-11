#include "condition_impl.hpp"
#include "oop/factory.hpp"
#include "utils/base_type_convert.hpp"
#include "utils/mexception.hpp"
#include <regex>

using namespace oop::factory;
using namespace utils::exception;

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

template <typename Comparator>
static bool compareValues(const DataField &data, const std::string &value,
                          Comparator comp) {
  auto val = utils::convert_string_to_number(value);

  return std::visit(
      [&](const auto &v) -> bool {
        using T = std::decay_t<decltype(v)>;

        if (data.type() != typeid(T)) {
          throw InvalidValueException{
              "The data types to be compared must be the same!"};
        }

        const auto &dataVal = std::any_cast<const T &>(data);
        return comp(dataVal, v);
      },
      val);
}

bool Equals::evaluate(DataField const &data, std::string const &value) const {
  return compareValues(data, value, std::equal_to<>());
}

bool NotEquals::evaluate(DataField const &data,
                         std::string const &value) const {
  return compareValues(data, value, std::not_equal_to<>());
}

bool GreaterThan::evaluate(DataField const &data,
                           std::string const &value) const {
  return compareValues(data, value, std::greater<>());
}

bool LessThan::evaluate(DataField const &data, std::string const &value) const {
  return compareValues(data, value, std::less<>());
}

bool GreaterThanOrEqual::evaluate(DataField const &data,
                                  std::string const &value) const {
  return compareValues(data, value, std::greater_equal<>());
}

bool LessThanOrEqual::evaluate(DataField const &data,
                               std::string const &value) const {
  return compareValues(data, value, std::less_equal<>());
}

bool IsOdd::evaluate(DataField const &data, std::string const &value) const {
  return compareValues(data, value, [](const auto &dataVal, const auto &) {
    if constexpr (std::is_integral_v<std::decay_t<decltype(dataVal)>>) {
      return dataVal % 2 != 0;
    } else {
      throw InvalidValueException{
          "IsOdd can only be applied to integral types!"};
    }
    return false;
  });
}

bool IsEven::evaluate(DataField const &data, std::string const &value) const {
  return compareValues(data, value, [](const auto &dataVal, const auto &) {
    if constexpr (std::is_integral_v<std::decay_t<decltype(dataVal)>>) {
      return dataVal % 2 == 0;
    } else {
      throw InvalidValueException{
          "IsEven can only be applied to integral types!"};
    }
    return false;
  });
}

RuleApplicatorRegistrar::RuleApplicatorRegistrar() {
  BasicLearningModuleRegister(RegexMatch);
  BasicLearningModuleRegister(RegexNotMatch);
  BasicLearningModuleRegister(Equals);
  BasicLearningModuleRegister(NotEquals);
  BasicLearningModuleRegister(GreaterThan);
  BasicLearningModuleRegister(LessThan);
  BasicLearningModuleRegister(GreaterThanOrEqual);
  BasicLearningModuleRegister(LessThanOrEqual);
  BasicLearningModuleRegister(IsOdd);
  BasicLearningModuleRegister(IsEven);
}
} // namespace rule_applicator
