#include "gflags/gflags.h"
#include "logger/logger.hpp"

DEFINE_int32(id, -1, "Specify the id.");
DEFINE_string(name, "", "Specify the name.");

DEFINE_string(configs, "",
              "Configuration of all algorithms that need to be started");

int main(int argc, char **argv) {

  BasicLearningLoggerInit(true, true, true, true);
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
//   if (FLAGS_help)
//     BASIC_LOGGER_INFO(gflags::ProgramInvocationShortName());
  BASIC_LOGGER_INFO(FLAGS_id);
  BASIC_LOGGER_INFO(FLAGS_name);
  gflags::ShutDownCommandLineFlags();

  BasicLearningLoggerDrop();

  return 0;
}
