/**
 * @file frame_infer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __NCNN_INFERENCE_FRAME__DET_HPP_
#define __NCNN_INFERENCE_FRAME__DET_HPP_

#include "dnn_infer.hpp"
#include "infer_types.hpp"
#include <memory>

namespace infer::dnn {
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
} // namespace infer::dnn
#endif