#include "params_center.hpp"
#include <chrono>
#include <cstddef>
#include <iostream>

using namespace features;

int main(int argc, char **argv) {
  // 创建参数中心实例
  ModuleParameterCenter param_center;

  // 创建不同类型的参数
  StreamBase stream_params{1920, 1080, "my://uri"};

  AlarmBase base{"./output", 10, 0.8f};
  InferInterval interval{std::chrono::seconds{3}};
  WithoutHelmetMonitor helmet_params{base, interval};

  // 存储 StreamBase 类型的参数
  param_center.setParams(stream_params);

  // 存储 WithoutHelmetMonitor 类型的参数
  param_center.setParams(helmet_params);

  // 访问参数
  param_center.visitParams([](auto &&params) {
    // 对于不同类型的参数，可以根据其类型进行不同的处理
    using T = std::decay_t<decltype(params)>; // 获取参数的实际类型
    if constexpr (std::is_same_v<T, StreamBase>) {
      std::cout << "StreamBase: " << params.uri << "\n";
    } else if constexpr (std::is_same_v<T, WithoutHelmetMonitor>) {
      std::cout << "WithoutHelmetMonitor: " << params.outputDir << "\n";
    } else {
      std::cout << "Unknown params\n";
    }
  });

  return 0;
}