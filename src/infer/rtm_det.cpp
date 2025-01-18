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

#include "rtm_det.hpp"
#include "infer_types.hpp"
#include "logger/logger.hpp"

namespace infer::dnn {
InferErrorCode RTMDetInference::infer(AlgoInput &input, AlgoOutput &output) {
  // Get input parameters
  auto *frameInput = input.getParams<FrameInput>();
  if (!frameInput) {
    LOGGER_ERROR("Invalid input parameters");
    return InferErrorCode::INFER_INPUT_ERROR;
  }

  try {

    // create infer engine
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in = preprocess(frameInput->image);

    ex.input(inputNames[0].c_str(), in);

    int numOutputs = outputNames.size();
    ncnn::Mat clsPred, detPred;
    ex.extract(outputNames[0].c_str(), clsPred);
    ex.extract(outputNames[1].c_str(), detPred);

    DetRet detections = postprocess(clsPred, detPred);
    output.setParams(detections);

    return InferErrorCode::SUCCESS;
  } catch (const std::exception &e) {
    LOGGER_ERROR("Inference failed: {}", e.what());
    return InferErrorCode::INFER_FAILED;
  }
}

ncnn::Mat RTMDetInference::preprocess(const cv::Mat &image) {
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(
      image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows,
      params->inputShape.w, params->inputShape.h);

  in.substract_mean_normalize(params->preprocessArg.meanVals.data(),
                              params->preprocessArg.normVals.data());
  return in;
}

DetRet RTMDetInference::postprocess(const ncnn::Mat &clsPred,
                                    const ncnn::Mat &detPred) {
  // TODO: implement postprocessing
  DetRet detections;
  return detections;
}

}; // namespace infer::dnn
