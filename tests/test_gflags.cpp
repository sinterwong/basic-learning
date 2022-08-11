#include "gflags/gflags.h"
#include "logger/logger.hpp"

DEFINE_int32(host_id, -1, "Specify device id.");
DEFINE_int32(camera_id, -1,
             "Specify the camera id, it is a unique identity of camera.");
DEFINE_int32(height, -1, "Specify video height.");
DEFINE_int32(width, -1, "Specify video width.");
DEFINE_int32(place, -1, "Specify video place.");
DEFINE_string(uri, "", "Specify the uri to run the camera.");
DEFINE_string(result_url, "", "Specify the url to send the results.");
DEFINE_string(codec, "h264", "Specify the video decoding mode.");
DEFINE_string(model_dir, "", "Specify the dir to models.");

DEFINE_string(configs, "",
              "Configuration of all algorithms that need to be started");

int main(int argc, char **argv) {

  FlowEngineLoggerInit(true, true, true, true);
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
//   if (FLAGS_help)
//     FLOWENGINE_LOGGER_INFO(gflags::ProgramInvocationShortName());
  FLOWENGINE_LOGGER_INFO(FLAGS_model_dir);
  FLOWENGINE_LOGGER_INFO(FLAGS_uri);
  FLOWENGINE_LOGGER_INFO(FLAGS_codec);
  gflags::ShutDownCommandLineFlags();

  FlowEngineLoggerDrop();

  return 0;
}
