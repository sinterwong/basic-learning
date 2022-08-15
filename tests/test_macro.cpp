

#define REGISTER_TENSORRT_PLUGIN(name)                                         \
  static nvinfer1::PluginRegistrar<name> pluginRegistrar##name {}

int main(int argc, char **argv) {
  int a = 0;
  return 0;
}