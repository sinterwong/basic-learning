#ifndef __SEGMENT_POST_PROCESSOR_HPP_
#define __SEGMENT_POST_PROCESSOR_HPP_

#include "measure_types.hpp"

namespace cv_process {
class SegPostProcessor {
public:
  SegPostProcessor(double min_cc_area_ratio = 0.3,
                   double min_plaque_area_ratio = 0.2, int min_plaque_area = 20,
                   double longitudinal_section_aspect_ratio = 2.0,
                   double longitudinal_section_eccentricity = 0.5);

  MeasureInput process(const SegmentResult &seg_ret) const;

  static std::vector<DetResult>
  extractBboxes(const cv::Mat &mask, const cv::Rect &roi, int plaque_label = 3);

private:
  double min_cc_area_ratio_;
  double min_plaque_area_ratio_;
  int min_plaque_area_;
  double longitudinal_section_aspect_ratio_;
  double longitudinal_section_eccentricity_;

  std::pair<std::vector<std::vector<cv::Point>>, std::vector<double>>
  findAndFilterContours(const cv::Mat &mask,
                        int contour_mode = cv::RETR_EXTERNAL,
                        int contour_method = cv::CHAIN_APPROX_NONE,
                        double area_ratio_th = 0.1, int area_th = -1) const;

  std::vector<std::vector<cv::Point>>
  shiftAndScaleContours(const std::vector<std::vector<cv::Point>> &contours,
                        double x_shift, double y_shift, double x_scale,
                        double y_scale, double x_roi_shift,
                        double y_roi_shift) const;

  bool isLongitudinalSection(const std::vector<cv::Point> &contour) const;

  static double
  getBoundingRectAspectRatio(const std::vector<cv::Point> &contour);

  static double getEccentricity(const std::vector<cv::Point> &contour);

  static int calculateFrameScanType(const std::vector<int> &scan_type_labels,
                                    const std::vector<double> &cc_areas);

  std::vector<std::vector<cv::Point>>
  processLumenContours(const cv::Mat &seg_crop, const cv::Rect &bbox,
                       const SegmentResult &seg_ret, int seg_w,
                       int seg_h) const;

  std::vector<std::vector<cv::Point>>
  processPlaqueContours(const cv::Mat &seg_crop, const cv::Rect &bbox,
                        const SegmentResult &seg_ret, int seg_w,
                        int seg_h) const;
};
} // namespace cv_process
#endif
