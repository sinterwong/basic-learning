#include "gflags/gflags.h"
#include "logger/logger.hpp"

DEFINE_int32(id, -1, "Specify the id.");
DEFINE_string(name, "", "Specify the name.");

DEFINE_string(configs, "",
              "Configuration of all algorithms that need to be started");

int main(int argc, char **argv) {

  FlowEngineLoggerInit(true, true, true, true);
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
//   if (FLAGS_help)
//     FLOWENGINE_LOGGER_INFO(gflags::ProgramInvocationShortName());
  FLOWENGINE_LOGGER_INFO(FLAGS_id);
  FLOWENGINE_LOGGER_INFO(FLAGS_name);
  gflags::ShutDownCommandLineFlags();

  FlowEngineLoggerDrop();

  return 0;
}
