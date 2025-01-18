/**
 * @file dnn_infer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __NCNN_INFERENCE_HPP_
#define __NCNN_INFERENCE_HPP_

#include "infer_types.hpp"
#include <memory>
#include <ncnn/net.h>

namespace infer::dnn {
class AlgoInference {
public:
  AlgoInference(const AlgoParamBase &param)
      : params(std::make_unique<AlgoParamBase>(param)) {}

  ~AlgoInference() {}

  virtual InferErrorCode initialize();

  virtual InferErrorCode infer(AlgoInput &input, AlgoOutput &output) = 0;

  virtual const ModelInfo &getModelInfo();

  virtual InferErrorCode terminate();

  virtual void prettyPrintModelInfos();

protected:
  std::unique_ptr<AlgoParamBase> params;
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;
  std::shared_ptr<ModelInfo> modelInfo;

  // infer engine TODO: How to decouple?
  ncnn::Net net;
  std::vector<ncnn::Mat> inputTensors;
  std::vector<ncnn::Mat> outputTensors;
};
} // namespace infer::dnn
#endif
