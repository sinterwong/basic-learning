/**
 * @file params_center.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-03-01
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <chrono>
#include <iostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#ifndef _FEATURES_PARAMETERS_CENTER_HPP_
#define _FEATURES_PARAMETERS_CENTER_HPP_

namespace template_mp {

using namespace std::chrono_literals;
using svector = std::vector<std::string>;

struct StreamBase {
  int width;
  int height;
  std::string uri;
};

struct AlarmBase {
  std::string outputDir;
  int videDuration;
  float threshold;
};

struct InferInterval {
  std::chrono::seconds interval{3}; // 间隔时间
};

struct WithoutHelmetMonitor : public AlarmBase, public InferInterval {
  WithoutHelmetMonitor(AlarmBase const &alarmBase,
                       InferInterval const &inferInterval)
      : AlarmBase(alarmBase), InferInterval(inferInterval) {}
};

struct SmokingMonitor : public AlarmBase {
  SmokingMonitor(AlarmBase const &alarmBase) : AlarmBase(alarmBase) {}
};

struct ExtinguisherMonitor : public InferInterval {
  ExtinguisherMonitor(InferInterval const &inferInterval)
      : InferInterval(inferInterval) {}
};

class ModuleParameterCenter {
public:
  // 将所有参数类型存储在一个 std::variant 中
  using Params = std::variant<StreamBase, WithoutHelmetMonitor, SmokingMonitor,
                              ExtinguisherMonitor>;

  // 设置参数
  template <typename T> void setParams(T params) {
    params_ = std::move(params);
  }

  // 访问参数
  template <typename Func> void visitParams(Func &&func) {
    std::visit([&](auto &&params) { std::forward<Func>(func)(params); },
               params_);
  }

private:
  Params params_;
};

// 模板特化时需要将特化定义放在命名空间作用域内，或者房子全局作用域内，不能放在类作用域内。
// 类作用域内的定义在编译器编译期间是不可见的，类作用域是在其声明的命名空间内定义的，但是他的成员函数和成员变量只有在类被实例化后才会被编译器看到。
template <> inline void ModuleParameterCenter::setParams(StreamBase params) {
  params_ = std::move(params);
  std::cout << "hello" << std::endl;
}

} // namespace template_mp
#endif // _FLOWENGINE_COMMON_CONFIG_HPP_