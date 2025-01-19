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

#include "frame_infer.hpp"
#include "infer_types.hpp"
#include "logger/logger.hpp"

namespace infer::dnn {

std::vector<std::pair<std::string, ncnn::Mat>>
FrameInference::preprocess(AlgoInput &input) const {

  std::vector<std::pair<std::string, ncnn::Mat>> ret;
  // Get input parameters
  auto *frameInput = input.getParams<FrameInput>();
  if (!frameInput) {
    LOGGER_ERROR("Invalid input parameters");
    throw std::runtime_error("Invalid input parameters");
  }
  const cv::Mat &image = frameInput->image;
  const auto &args = frameInput->args;

  // TODO: support roi and padding
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(
      image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows,
      params->inputShape.w, params->inputShape.h);

  in.substract_mean_normalize(args.meanVals.data(), args.normVals.data());
  ret.emplace_back(inputNames[0], in);
  return ret;
}

}; // namespace infer::dnn
