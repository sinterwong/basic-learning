#include "params_center.hpp"
#include <chrono>
#include <gtest/gtest.h>

using namespace template_mp;

TEST(TestParamsCenter, TestParamsCenter) {
  ModuleParameterCenter param_center;

  StreamBase stream_params{1920, 1080, "my://uri"};

  AlarmBase base{"./output", 10, 0.8f};
  InferInterval interval{std::chrono::seconds{3}};
  WithoutHelmetMonitor helmet_params{base, interval};

  SmokingMonitor smoking_params{base};
  ExtinguisherMonitor extinguisher_params{interval};

  param_center.setParams(stream_params);
  param_center.visitParams([](auto &params) {
    using T = std::decay_t<decltype(params)>;
    if constexpr (std::is_same_v<T, StreamBase>) {
      std::cout << params.width << " " << params.height << " " << params.uri
                << std::endl;
    } else if constexpr (std::is_same_v<T, WithoutHelmetMonitor>) {
      std::cout << params.outputDir << " " << params.videDuration << " "
                << params.threshold << " " << params.interval.count()
                << std::endl;
    } else if constexpr (std::is_same_v<T, SmokingMonitor>) {
      std::cout << params.outputDir << " " << params.videDuration << " "
                << params.threshold << std::endl;
    } else if constexpr (std::is_same_v<T, ExtinguisherMonitor>) {
      std::cout << params.interval.count() << std::endl;
    }
  });
}
