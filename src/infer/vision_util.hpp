/**
 * @file vision_util.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_VISION_UTILS_HPP_
#define __INFERENCE_VISION_UTILS_HPP_

#include "infer_types.hpp"

namespace infer::utils {

std::pair<float, float> scaleRatio(Shape const &originShape,
                                   Shape const &inputShape, bool isScale);

std::vector<BBox> NMS(const std::vector<BBox> &results, float nmsThre,
                      float confThre);

Shape escaleResizeWithPad(const cv::Mat &src, cv::Mat &dst, int targetWidth,
                          int targetHeight, const cv::Scalar &pad);
} // namespace infer::utils
#endif