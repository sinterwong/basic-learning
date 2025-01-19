/**
 * @file yoloDet.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_VISION_YOLOV11_DETECTION_HPP_
#define __INFERENCE_VISION_YOLOV11_DETECTION_HPP_

#include "infer_types.hpp"
#include "vision.hpp"
namespace infer::dnn::vision {
class Yolov11Det : public Vision {
public:
  explicit Yolov11Det(AlgoPostprocParams &params) : mParams(params) {}

  virtual bool processOutput(const ModelOutput &, const FramePreprocessArg &,
                             AlgoOutput &) override;

private:
  std::vector<BBox> processRawOutput(const cv::Mat &transposedOutput,
                                     const Shape &inputShape,
                                     const FramePreprocessArg &args,
                                     int numClasses);

private:
  AlgoPostprocParams mParams;
};
} // namespace infer::dnn::vision

#endif
