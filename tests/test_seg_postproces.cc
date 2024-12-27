#include "measure_types.hpp"
#include "seg_ret_postprocess.hpp"
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <opencv2/core/types.hpp>
#include <vector>

using namespace cv_process;
namespace fs = std::filesystem;

TEST(SegPostProcessTest, SegMaskToContours) {
  fs::path dataDir = fs::path("data/measure/mask");

  std::string maskPath = dataDir / "image.png";

  cv::Mat seg_mask = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);
  if (seg_mask.empty()) {
    return;
  }

  cv::Mat input_mask;
  cv::resize(seg_mask, input_mask, cv::Size(960, 960));

  SegPostProcessor processor(0.3, 0.2, 20, 2.0, 0.5);

  SegmentResult seg_ret;
  seg_ret.mask = input_mask;
  seg_ret.roi = cv::Rect{0, 0, seg_mask.cols, seg_mask.rows};

  MeasureInput result = processor.process(seg_ret);

  // visualize info
  cv::Mat vis_img = seg_mask.clone() * 255;
  cv::cvtColor(vis_img, vis_img, cv::COLOR_GRAY2BGR);
  for (const auto &pair : result.contour_pairs) {
    cv::polylines(vis_img, pair.first, true, cv::Scalar(0, 0, 255), 2);
    cv::polylines(vis_img, pair.second, true, cv::Scalar(255, 0, 0), 2);
  }

  auto bboxes = processor.extractBboxes(seg_mask, seg_ret.roi);
  for (const auto &bbox : bboxes) {
    cv::rectangle(vis_img, bbox.bbox, cv::Scalar(255, 255, 0), 2);
  }
  cv::imwrite("vis_contours.png", vis_img);
  ASSERT_GE(result.contour_pairs.size(), 0);
}
