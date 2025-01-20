/**
 * @file types.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-11-22
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __INFERENCE_TYPES_HPP__
#define __INFERENCE_TYPES_HPP__

#include <opencv2/opencv.hpp>
#include <string>
#include <variant>

namespace infer {

enum class InferErrorCode : int32_t {
  SUCCESS = 0,

  // init error
  INIT_FAILED = 100,
  INIT_CONFIG_FAILED = 101,
  INIT_MODEL_LOAD_FAILED = 102,
  INIT_DEVICE_FAILED = 103,

  // infer error
  INFER_FAILED = 200,
  INFER_INPUT_ERROR = 201,
  INFER_OUTPUT_ERROR = 202,
  INFER_DEVICE_ERROR = 203,

  // release error
  TERMINATE_FAILED = 300
};

enum class DeviceType { CPU = 0, GPU = 1 };

struct Shape {
  int w;
  int h;
};

struct FramePreprocessArg {
  cv::Rect roi;
  std::vector<float> meanVals;
  std::vector<float> normVals;
  Shape originShape;

  bool isEqualScale;
  cv::Scalar pad = {0, 0, 0};
  int topPad = 0;
  int leftPad = 0;
};

struct FrameInput {
  cv::Mat image;
  FramePreprocessArg args;
};

struct ModelInfo {
  std::string name;

  struct InputInfo {
    std::string name;
    std::vector<int64_t> shape;
  };

  struct OutputInfo {
    std::string name;
    std::vector<int64_t> shape;
  };

  std::vector<InputInfo> inputs;
  std::vector<OutputInfo> outputs;
};

// Algo input
class AlgoInput {
public:
  using Params = std::variant<std::monostate, FrameInput>;

  template <typename T> void setParams(T params) {
    params_ = std::move(params);
  }

  template <typename Func> void visitParams(Func &&func) {
    std::visit([&](auto &&params) { std::forward<Func>(func)(params); },
               params_);
  }

  template <typename T> T *getParams() { return std::get_if<T>(&params_); }

private:
  Params params_;
};

// Model output(after infering, before postprocess)
struct ModelOutput {
  std::vector<std::vector<float>> outputs;
  std::vector<std::vector<int>> outputShapes;
};

// Algo output
struct BBox {
  cv::Rect rect;
  float score;
  int label;
};

struct ClsRet {
  float score;
  int label;
};

struct DetRet {
  std::vector<BBox> bboxes;
};

class AlgoOutput {
public:
  using Params = std::variant<std::monostate, ClsRet, DetRet>;

  template <typename T> void setParams(T params) {
    params_ = std::move(params);
  }

  template <typename Func> void visitParams(Func &&func) {
    std::visit([&](auto &&params) { std::forward<Func>(func)(params); },
               params_);
  }

  template <typename T> T *getParams() { return std::get_if<T>(&params_); }

private:
  Params params_;
};

// Post-process Params
struct AnchorDetParams {
  float condThre;
  float nmsThre;
  Shape inputShape;
};

class AlgoPostprocParams {
public:
  using Params = std::variant<std::monostate, AnchorDetParams>;

  template <typename T> void setParams(T params) {
    params_ = std::move(params);
  }

  template <typename Func> void visitParams(Func &&func) {
    std::visit([&](auto &&params) { std::forward<Func>(func)(params); },
               params_);
  }

  template <typename T> T *getParams() { return std::get_if<T>(&params_); }

private:
  Params params_;
};

// Infer Params
struct InferParamBase {
  std::string name;
  std::string modelPath;
  DeviceType deviceType;
};

struct FrameInferParam : public InferParamBase {
  Shape inputShape;
};
} // namespace infer
#endif
