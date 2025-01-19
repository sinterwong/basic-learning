/**
 * @file vision.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_VISION_HPP_
#define __INFERENCE_VISION_HPP_

#include "infer_types.hpp"

namespace infer::dnn::vision {
class Vision {
public:
  explicit Vision() {}

  virtual ~Vision(){};

  virtual bool processOutput(const ModelOutput &, const FramePreprocessArg &,
                             AlgoOutput &) = 0;
};
} // namespace infer::dnn::vision

#endif