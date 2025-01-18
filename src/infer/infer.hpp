/**
 * @file infer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_HPP_
#define __INFERENCE_HPP_

#include "infer_types.hpp"
#include <memory>

namespace infer::dnn {
class Inference {
public:
  Inference() = default;

  virtual ~Inference() {}

  virtual InferErrorCode initialize() = 0;

  virtual InferErrorCode infer(AlgoInput &input, ModelOutput &modelOutput) = 0;

  virtual InferErrorCode terminate() = 0;

  virtual const ModelInfo &getModelInfo() = 0;

  virtual void prettyPrintModelInfos();

protected:
  std::shared_ptr<ModelInfo> modelInfo;
};
} // namespace infer::dnn
#endif
