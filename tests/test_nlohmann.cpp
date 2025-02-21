#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

TEST(NlohmannTest, Normal) {
  json j = {
      {"pi", 3.141},
      {"happy", true},
      {"name", "Niels"},
      {"nothing", nullptr},
      {"answer", {{"everything", 42}}},
  };
  std::string s = j.dump();
  ASSERT_NE(s.find("3.141"), std::string::npos);
  ASSERT_NE(s.find("true"), std::string::npos);
  ASSERT_NE(s.find("Niels"), std::string::npos);
  ASSERT_NE(s.find("null"), std::string::npos);
  ASSERT_NE(s.find("42"), std::string::npos);
}
