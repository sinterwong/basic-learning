#ifndef __NCNN_INFERENCE_RTM_DET_HPP_
#define __NCNN_INFERENCE_RTM_DET_HPP_

#include "dnn_infer.hpp"
#include "infer_types.hpp"
#include <memory>

namespace infer::dnn {
class RTMDetInference : public AlgoInference {
public:
  explicit RTMDetInference(const RTMDetParam &param)
      : AlgoInference(param), params(std::make_unique<RTMDetParam>(param)) {}

  InferErrorCode infer(AlgoInput &input, AlgoOutput &output) override;

private:
  ncnn::Mat preprocess(const cv::Mat &image);

  DetRet postprocess(const ncnn::Mat &clsPred, const ncnn::Mat &detPred);

private:
  std::unique_ptr<RTMDetParam> params;
};
} // namespace infer::dnn
#endif