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
#include "infer_types.hpp"

namespace infer::dnn::vision {
class Vision {
public:
  explicit Vision() {}

  virtual ~Vision(){};

  virtual bool processOutput(const ModelOutput &, AlgoOutput &) const = 0;

protected:
};
} // namespace infer::dnn::vision
