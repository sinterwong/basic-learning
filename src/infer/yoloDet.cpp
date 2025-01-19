/**
 * @file yoloDet.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "yoloDet.hpp"
#include "infer_types.hpp"
#include "logger/logger.hpp"
#include "vision_util.hpp"

namespace infer::dnn::vision {
bool Yolov11Det::processOutput(const ModelOutput &modelOutput,
                               const FramePreprocessArg &args,
                               AlgoOutput &algoOutput) {
  if (modelOutput.outputs.empty()) {
    return false;
  }

  auto params = mParams.getParams<YoloDetParams>();
  if (params == nullptr) {
    LOGGER_ERROR("YoloDetParams params is nullptr");
    throw std::runtime_error("YoloDetParams params is nullptr");
  }

  const auto &outputShapes = modelOutput.outputShapes;
  const auto &inputShape = params->inputShape;
  const auto &outputs = modelOutput.outputs;

  // just one output
  auto output = outputs.at(0);

  std::vector<int> outputShape = outputShapes.at(0);
  int signalResultNum = outputShape.at(outputShape.size() - 2);
  int strideNum = outputShape.at(outputShape.size() - 1);

  cv::Mat rawData(strideNum, signalResultNum, CV_32F);
  cv::transpose(
      cv::Mat(signalResultNum, strideNum, CV_32F, (void *)output.data()),
      rawData);
  std::vector<BBox> results =
      processRawOutput(rawData, inputShape, args, signalResultNum - 4);

  DetRet detRet;
  detRet.bboxes = utils::NMS(results, params->nmsThre, params->condThre);
  algoOutput.setParams(detRet);
  return true;
}

std::vector<BBox> Yolov11Det::processRawOutput(const cv::Mat &transposedOutput,
                                               const Shape &inputShape,
                                               const FramePreprocessArg &args,
                                               int numClasses) {
  std::vector<BBox> results;

  auto params = mParams.getParams<YoloDetParams>();
  if (params == nullptr) {
    LOGGER_ERROR("YoloDetParams params is nullptr");
    throw std::runtime_error("YoloDetParams params is nullptr");
  }
  Shape originShape;
  if (args.roi.area() > 0) {
    originShape.w = args.roi.width;
    originShape.h = args.roi.height;
  } else {
    originShape = args.originShape;
  }
  auto [scaleX, scaleY] =
      utils::scaleRatio(originShape, inputShape, args.isEqualScale);

  for (int i = 0; i < transposedOutput.rows; ++i) {
    const float *data = transposedOutput.ptr<float>(i);

    cv::Mat scores(1, numClasses, CV_32F, (void *)(data + 4));
    cv::Point classIdPoint;
    double score;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);

    if (score > params->condThre) {
      BBox result;
      result.score = score;
      result.label = classIdPoint.x;

      float x = data[0];
      float y = data[1];
      float w = data[2];
      float h = data[3];

      x = x - 0.5 * w;
      y = y - 0.5 * h;

      if (args.isEqualScale) {
        x = (x - args.leftPad) / scaleX;
        y = (y - args.topPad) / scaleY;
      } else {
        x = x / scaleX;
        y = y / scaleY;
      }
      w = w / scaleX;
      h = h / scaleY;
      x += args.roi.x;
      y += args.roi.y;
      result.rect = {static_cast<int>(x), static_cast<int>(y),
                     static_cast<int>(w), static_cast<int>(h)};
      results.push_back(result);
    }
  }

  return results;
}
} // namespace infer::dnn::vision
