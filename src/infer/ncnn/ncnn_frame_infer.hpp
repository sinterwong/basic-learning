#ifndef __NCNN_INFERENCE_FRAME__DET_HPP_
#define __NCNN_INFERENCE_FRAME__DET_HPP_

#include "infer_types.hpp"
#include "ncnn_dnn_infer.hpp"
#include <memory>

namespace infer::dnn::ncnn_infer {
class FrameInference : public AlgoInference {
public:
  explicit FrameInference(const FrameInferParam &param)
      : AlgoInference(param), params(std::make_unique<FrameInferParam>(param)) {
  }

private:
  std::vector<std::pair<std::string, ncnn::Mat>>
  preprocess(AlgoInput &input) const override;

private:
  std::unique_ptr<FrameInferParam> params;
};
} // namespace infer::dnn::ncnn_infer
#endif