#include <gflags/gflags.h>
#include <iostream>

// Command flags
DEFINE_string(param, "", "cli param");

// Validation functions
static bool ValidateParam(const char *flagname, const std::string &value) {
  if (value != "aaa" && value != "bbb") {
    std::cout << "Invalid param. Must be 'aaa' or 'bbb'" << std::endl;
    return false;
  }
  return true;
}

// Register validators
DEFINE_validator(param, &ValidateParam);

void printUsage() {
  std::cout << "Usage: one_cli_tool --param=<value>" << std::endl;
  std::cout << "  --param: cli param, must be 'aaa' or 'bbb'" << std::endl;
}

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("One CLI Tool");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_param.empty()) {
    printUsage();
    return 1;
  }
  return 0;
}