#ifndef __CV_PROCESS_MEASURE_TYPE_HPP_
#define __CV_PROCESS_MEASURE_TYPE_HPP_
#include <opencv2/opencv.hpp>
#include <utility>

namespace cv_process {

using Contour = std::vector<cv::Point>;

struct MeasureRet {
  float thickness = 0.f;
  float length = 0.f;
  std::pair<cv::Point, cv::Point> thichness_line;
  std::pair<cv::Point, cv::Point> length_line;
};

struct MeasureInput {
  int scan_type;
  std::vector<std::pair<Contour, Contour>> contour_pairs;
};

struct SegmentResult {
  cv::Mat mask;

  // mask可能是roi之后的模型尺寸，因此提供roi信息用于可能的尺寸还原
  cv::Rect roi;
};

struct DetResult {
  cv::Rect bbox;
  float score;
  int label;
};

} // namespace cv_process
#endif
