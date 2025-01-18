/**
 * @file dnn_infer.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "dnn_infer.hpp"
#include "infer_types.hpp"
#include "logger/logger.hpp"

namespace infer::dnn {
InferErrorCode AlgoInference::initialize() {
  if (net.load_param(params->paramPath.c_str()) != 0) {
    LOGGER_ERROR("Failed to load param: {}", params->paramPath);
    return InferErrorCode::INIT_MODEL_LOAD_FAILED;
  }
  if (net.load_model(params->modelPath.c_str()) != 0) {
    LOGGER_ERROR("Failed to load model: {}", params->modelPath.c_str());
    return InferErrorCode::INIT_MODEL_LOAD_FAILED;
  }

  // get inputNames and outputNames
  for (auto const &inName : net.input_names()) {
    inputNames.push_back(inName);
  }
  for (auto const &outName : net.output_names()) {
    outputNames.push_back(outName);
  }
  return InferErrorCode::SUCCESS;
}

const ModelInfo &AlgoInference::getModelInfo() {
  if (modelInfo)
    return *modelInfo;
  if (!params) {
    LOGGER_ERROR("Invalid algorithm parameters");
    return *modelInfo;
  }

  modelInfo = std::make_shared<ModelInfo>();
  modelInfo->name = params->name;
  for (const auto &inputName : inputNames) {
    ncnn::Mat inputTensor;
    modelInfo->inputs.push_back({inputName, {}});
  }
  for (const auto &outputName : outputNames) {
    ncnn::Mat outputTensor;
    modelInfo->outputs.push_back({outputName, {}});
  }
  return *modelInfo;
}

InferErrorCode AlgoInference::terminate() {
  net.clear();
  inputTensors.clear();
  outputTensors.clear();
  modelInfo.reset();
  return InferErrorCode::SUCCESS;
}

void AlgoInference::prettyPrintModelInfos() {
  if (!modelInfo) {
    getModelInfo();
    if (!modelInfo) {
      return;
    }
  }
  std::cout << "Model Name: " << modelInfo->name << std::endl;
  std::cout << "Inputs:" << std::endl;
  for (const auto &input : modelInfo->inputs) {
    std::cout << "  Name: " << input.name << ", Shape: ";
    for (int64_t dim : input.shape) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Outputs:" << std::endl;
  for (const auto &output : modelInfo->outputs) {
    std::cout << "  Name: " << output.name << ", Shape: ";
    for (int64_t dim : output.shape) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
  }
}
}; // namespace infer::dnn
