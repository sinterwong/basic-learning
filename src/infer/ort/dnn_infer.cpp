/**
 * @file ort_dnn_infer.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-18
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "dnn_infer.hpp"
#include "logger/logger.hpp"
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <codecvt>
#endif

namespace infer::dnn {
InferErrorCode AlgoInference::initialize() {
  try {
    // create environment
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                     params->name.c_str());

    // session options
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // create session
#ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring wmodelPath = converter.from_bytes(mParams.modelPath);
    session = std::make_unique<Ort::Session>(*env, wmodelPath.c_str(),
                                             sessionOptions);
#else
    // Non-Windows platforms can use the original code
    session = std::make_unique<Ort::Session>(*env, params->modelPath.c_str(),
                                             sessionOptions);
#endif

    // create memory info
    memoryInfo = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    // get input info
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = session->GetInputCount();
    inputNames.resize(numInputNodes);
    inputShapes.resize(numInputNodes);

    for (size_t i = 0; i < numInputNodes; i++) {
      // get input name
      auto inputName = session->GetInputNameAllocated(i, allocator);
      inputNames[i] = inputName.get();

      // get input shape
      auto typeInfo = session->GetInputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
      inputShapes[i] = tensorInfo.GetShape();
    }

    // get output info
    size_t numOutputNodes = session->GetOutputCount();
    outputNames.resize(numOutputNodes);
    outputShapes.resize(numOutputNodes);

    for (size_t i = 0; i < numOutputNodes; i++) {
      // get output name
      auto outputName = session->GetOutputNameAllocated(i, allocator);
      outputNames[i] = outputName.get();

      // get output shape
      auto typeInfo = session->GetOutputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
      outputShapes[i] = tensorInfo.GetShape();
    }

    return InferErrorCode::SUCCESS;
  } catch (const Ort::Exception &e) {
    LOGGER_ERROR("ONNX Runtime error during initialization: {}", e.what());
    return InferErrorCode::INIT_MODEL_LOAD_FAILED;

  } catch (const std::exception &e) {
    LOGGER_ERROR("Error during initialization: {}", e.what());
    return InferErrorCode::INIT_FAILED;
  }
}

InferErrorCode AlgoInference::infer(AlgoInput &input,
                                    ModelOutput &modelOutput) {

  try {
    auto inputsDatas = preprocess(input);

    std::vector<const char *> inputNamesPtr;
    std::vector<const char *> outputNamesPtr;

    for (const auto &name : inputNames) {
      inputNamesPtr.push_back(name.c_str());
    }
    for (const auto &name : outputNames) {
      outputNamesPtr.push_back(name.c_str());
    }

    std::vector<Ort::Value> inputs;
    for (int i = 0; i < inputsDatas.size(); ++i) {
      // create tensor
      inputs.emplace_back(Ort::Value::CreateTensor<float>(
          *memoryInfo, inputsDatas[i].data(), inputsDatas[i].size(),
          inputShapes[i].data(), inputShapes[i].size()));
    }

    auto outputs = session->Run(Ort::RunOptions{nullptr}, inputNamesPtr.data(),
                                inputs.data(), inputs.size(),
                                outputNamesPtr.data(), outputNames.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
      auto &output = outputs[i];
      std::vector<float> outputData(
          output.GetTensorTypeAndShapeInfo().GetElementCount());
      memcpy(outputData.data(), output.GetTensorMutableData<float>(),
             output.GetTensorTypeAndShapeInfo().GetElementCount() *
                 sizeof(float));
      modelOutput.outputs.emplace_back(outputData);
      std::vector<int> outputShape;
      for (int64_t dim : output.GetTensorTypeAndShapeInfo().GetShape()) {
        outputShape.push_back(dim);
      }
      modelOutput.outputShapes.push_back(outputShape);
    }
    return InferErrorCode::SUCCESS;
  } catch (const Ort::Exception &e) {
    LOGGER_ERROR("ONNX Runtime error during inference: {}", e.what());
    return InferErrorCode::INFER_FAILED;
  } catch (const std::exception &e) {
    LOGGER_ERROR("Error during inference: {}", e.what());
    return InferErrorCode::INFER_FAILED;
  }
}

InferErrorCode AlgoInference::terminate() {
  try {
    session.reset();
    env.reset();
    memoryInfo.reset();

    inputNames.clear();
    inputShapes.clear();
    outputNames.clear();
    outputShapes.clear();

    return InferErrorCode::SUCCESS;
  } catch (const std::exception &e) {
    LOGGER_ERROR("Error during termination: {}", e.what());
    return InferErrorCode::TERMINATE_FAILED;
  }
}

const ModelInfo &AlgoInference::getModelInfo() {
  if (modelInfo)
    return *modelInfo;

  modelInfo = std::make_shared<ModelInfo>();

  modelInfo->name = params->name;
  if (!session) {
    LOGGER_ERROR("Session is not initialized");
    return *modelInfo;
  }
  try {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = session->GetInputCount();
    modelInfo->inputs.resize(numInputNodes);
    for (size_t i = 0; i < numInputNodes; i++) {
      auto inputName = session->GetInputNameAllocated(i, allocator);
      modelInfo->inputs[i].name = inputName.get();

      auto typeInfo = session->GetInputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

      modelInfo->inputs[i].shape = tensorInfo.GetShape();

      size_t numOutputNodes = session->GetOutputCount();
      modelInfo->outputs.resize(numOutputNodes);

      for (size_t i = 0; i < numOutputNodes; i++) {
        auto outputName = session->GetOutputNameAllocated(i, allocator);
        modelInfo->outputs[i].name = outputName.get();
        auto typeInfo = session->GetOutputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        modelInfo->outputs[i].shape = tensorInfo.GetShape();
      }
    }
  } catch (const Ort::Exception &e) {
    LOGGER_ERROR("ONNX Runtime error during getting model info: {}", e.what());
  } catch (const std::exception &e) {
    LOGGER_ERROR("Error during getting model info: {}", e.what());
  }
  return *modelInfo;
}
}; // namespace infer::dnn
