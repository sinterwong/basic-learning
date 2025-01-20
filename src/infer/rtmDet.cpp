/**
 * @file rtmDet.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-20
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "rtmDet.hpp"
#include "infer_types.hpp"
#include "logger/logger.hpp"
#include "vision_util.hpp"

namespace infer::dnn::vision {
bool RTMDet::processOutput(const ModelOutput &modelOutput,
                           const FramePreprocessArg &args,
                           AlgoOutput &algoOutput) {
  if (modelOutput.outputs.empty()) {
    return false;
  }

  auto params = mParams.getParams<AnchorDetParams>();
  if (params == nullptr) {
    LOGGER_ERROR("AnchorDetParams params is nullptr");
    throw std::runtime_error("AnchorDetParams params is nullptr");
  }

  const auto &outputShapes = modelOutput.outputShapes;
  const auto &inputShape = params->inputShape;
  const auto &outputs = modelOutput.outputs;

  // two output
  auto detPred = outputs.at(0);
  auto clsPred = outputs.at(1);

  std::vector<int> detOutShape = outputShapes.at(0);
  std::vector<int> clsOutShape = outputShapes.at(1);

  int numClasses = clsOutShape.at(clsOutShape.size() - 1);
  int anchorNum = detOutShape.at(detOutShape.size() - 2);

  Shape originShape;
  if (args.roi.area() > 0) {
    originShape.w = args.roi.width;
    originShape.h = args.roi.height;
  } else {
    originShape = args.originShape;
  }
  auto [scaleX, scaleY] =
      utils::scaleRatio(originShape, inputShape, args.isEqualScale);

  std::vector<BBox> results;
  for (int i = 0; i < anchorNum; ++i) {
    float *detData = (float *)(detPred.data() + i * 4);
    float *clsData = (float *)(clsPred.data() + i * numClasses);
    cv::Mat scores(1, numClasses, CV_32F, clsData);
    cv::Point classIdPoint;
    double score;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);
    if (score > params->condThre) {
      float x = detData[0];
      float y = detData[1];
      float w = detData[2] - x;
      float h = detData[3] - y;

      if (args.isEqualScale) {
        x = (x - args.leftPad) / scaleX;
        y = (y - args.topPad) / scaleY;
      } else {
        x = x / scaleX;
        y = y / scaleY;
      }
      w = w / scaleX;
      h = h / scaleY;
      BBox result;
      result.score = score;
      result.label = classIdPoint.x;
      x += args.roi.x;
      y += args.roi.y;

      result.rect = {static_cast<int>(x), static_cast<int>(y),
                     static_cast<int>(w), static_cast<int>(h)};
      results.emplace_back(result);
    }
  }

  DetRet detRet;
  detRet.bboxes = utils::NMS(results, params->nmsThre, params->condThre);
  algoOutput.setParams(detRet);
  return true;
}

} // namespace infer::dnn::vision
