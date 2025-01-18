/**
 * @file ort_dnn_infer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __ONNXRUNTIME_INFERENCE_H_
#define __ONNXRUNTIME_INFERENCE_H_

#include "infer.hpp"
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace infer::dnn::ort_infer {
class AlgoInference : public Inference {
public:
  AlgoInference(const AlgoParamBase &params)
      : params(std::make_unique<AlgoParamBase>(params)) {}

  virtual ~AlgoInference() {}

  virtual InferErrorCode initialize() override;

  virtual InferErrorCode infer(AlgoInput &input,
                               ModelOutput &modelOutput) override;

  virtual const ModelInfo &getModelInfo() override;

  virtual InferErrorCode terminate() override;

protected:
  virtual std::vector<Ort::Value> preprocess(AlgoInput &input) const = 0;

protected:
  std::unique_ptr<AlgoParamBase> params;
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;

  std::vector<std::vector<int64_t>> inputShape;
  std::vector<std::vector<int64_t>> outputShapes;

  // infer engine
  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::Session> session;
  std::unique_ptr<Ort::MemoryInfo> memoryInfo;
};
} // namespace infer::dnn::ort_infer
#endif
