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
#include "vision_util.hpp"
#include <opencv2/core/mat.hpp>

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

  int inputWidth = params->inputShape.w;
  int inputHeight = params->inputShape.h;

  const cv::Mat &image = frameInput->image;
  auto &args = frameInput->args;

  // crop roi
  cv::Mat croppedImage;
  if (args.roi.area() > 0) {
    croppedImage = image(args.roi).clone();
  } else {
    croppedImage = image;
  }

  ncnn::Mat in;
  if (args.isEqualScale) {
    cv::Mat resizedImage;
    auto padRet = utils::escaleResizeWithPad(
        croppedImage, resizedImage, inputHeight, inputHeight, args.pad);
    args.topPad = padRet.h;
    args.leftPad = padRet.w;
    in = ncnn::Mat::from_pixels(resizedImage.data, ncnn::Mat::PIXEL_RGB,
                                resizedImage.cols, resizedImage.rows);
  } else {
    in = ncnn::Mat::from_pixels_resize(
        image.data, ncnn::Mat::PIXEL_RGB, image.cols, image.rows,
        params->inputShape.w, params->inputShape.h);
  }
  std::vector<float> normVals;
  std::transform(args.normVals.begin(), args.normVals.end(),
                 std::back_inserter(normVals),
                 [](float val) { return 1.0f / val; });
  in.substract_mean_normalize(args.meanVals.data(), normVals.data());
  ret.emplace_back(inputNames[0], in);
  return ret;
}

}; // namespace infer::dnn
