#include "seg_ret_postprocess.hpp"
#include <cmath>
#include <numeric>
#include <vector>

namespace cv_process {
SegPostProcessor::SegPostProcessor(double min_cc_area_ratio,
                                   double min_plaque_area_ratio,
                                   int min_plaque_area,
                                   double longitudinal_section_aspect_ratio,
                                   double longitudinal_section_eccentricity)
    : min_cc_area_ratio_(min_cc_area_ratio),
      min_plaque_area_ratio_(min_plaque_area_ratio),
      min_plaque_area_(min_plaque_area),
      longitudinal_section_aspect_ratio_(longitudinal_section_aspect_ratio),
      longitudinal_section_eccentricity_(longitudinal_section_eccentricity) {}

MeasureInput SegPostProcessor::process(const SegmentResult &seg_ret) const {
  cv::Mat mask = seg_ret.mask;
  int ori_h = seg_ret.roi.height;
  int ori_w = seg_ret.roi.width;
  int seg_h = mask.rows;
  int seg_w = mask.cols;

  auto [cc_contours, cc_areas] = findAndFilterContours(
      mask, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, min_cc_area_ratio_);

  std::vector<int> scan_type_labels;
  std::vector<std::pair<std::vector<cv::Point>, std::vector<cv::Point>>>
      cc_plaque_lumen_contour_pairs;

  for (const auto &cc_contour : cc_contours) {
    int scan_type_label = isLongitudinalSection(cc_contour) ? 1 : 0;
    scan_type_labels.push_back(scan_type_label);

    cv::Rect bbox = cv::boundingRect(cc_contour);
    cv::Mat seg_crop = mask(bbox);

    auto lumen_contours =
        processLumenContours(seg_crop, bbox, seg_ret, seg_w, seg_h);
    auto plaque_contours =
        processPlaqueContours(seg_crop, bbox, seg_ret, seg_w, seg_h);

    for (const auto &lumen_contour : lumen_contours) {
      for (const auto &plaque_contour : plaque_contours) {
        cc_plaque_lumen_contour_pairs.emplace_back(plaque_contour,
                                                   lumen_contour);
      }
    }
  }

  MeasureInput result;
  result.scan_type = calculateFrameScanType(scan_type_labels, cc_areas);
  result.contour_pairs = cc_plaque_lumen_contour_pairs;
  return result;
}

std::vector<DetResult> SegPostProcessor::extractBboxes(const cv::Mat &mask,
                                                       const cv::Rect &roi,
                                                       int plaque_label) {
  int ori_w = roi.width;
  int ori_h = roi.height;
  std::vector<DetResult> plaque_ins_list;
  cv::Mat plaque_mask = (mask == plaque_label);
  std::vector<cv::Point> plaque_points;
  cv::findNonZero(plaque_mask, plaque_points);

  if (!plaque_points.empty()) {
    cv::Rect bbox = cv::boundingRect(plaque_points);

    float resize_ratio_x =
        static_cast<float>(ori_w) / static_cast<float>(mask.cols);
    float resize_ratio_y =
        static_cast<float>(ori_h) / static_cast<float>(mask.rows);

    bbox.x = static_cast<int>(bbox.x * resize_ratio_x) + roi.x;
    bbox.y = static_cast<int>(bbox.y * resize_ratio_y) + roi.y;
    bbox.width = static_cast<int>(bbox.width * resize_ratio_x);
    bbox.height = static_cast<int>(bbox.height * resize_ratio_y);

    plaque_ins_list.emplace_back(DetResult{bbox, 0.8f, plaque_label});
  }

  return plaque_ins_list;
}

std::pair<std::vector<std::vector<cv::Point>>, std::vector<double>>
SegPostProcessor::findAndFilterContours(const cv::Mat &mask, int contour_mode,
                                        int contour_method,
                                        double area_ratio_th,
                                        int area_th) const {
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(mask, contours, hierarchy, contour_mode, contour_method);

  std::vector<double> contour_areas;
  for (const auto &c : contours) {
    contour_areas.push_back(cv::contourArea(c));
  }

  if (contours.empty()) {
    return std::make_pair(std::vector<std::vector<cv::Point>>(),
                          std::vector<double>());
  }

  std::vector<int> valid_indices(contours.size());
  std::iota(valid_indices.begin(), valid_indices.end(), 0);

  if (area_ratio_th > 0 && area_ratio_th <= 1) {
    double total_area =
        std::accumulate(contour_areas.begin(), contour_areas.end(), 0.0);
    valid_indices.erase(
        std::remove_if(valid_indices.begin(), valid_indices.end(),
                       [&](int i) {
                         return contour_areas[i] < total_area * area_ratio_th;
                       }),
        valid_indices.end());
  }

  if (area_th >= 0) {
    valid_indices.erase(
        std::remove_if(valid_indices.begin(), valid_indices.end(),
                       [&](int i) { return contour_areas[i] < area_th; }),
        valid_indices.end());
  }

  std::vector<std::vector<cv::Point>> filtered_contours;
  std::vector<double> filtered_areas;
  for (int i : valid_indices) {
    filtered_contours.push_back(contours[i]);
    filtered_areas.push_back(contour_areas[i]);
  }

  return std::make_pair(filtered_contours, filtered_areas);
}

std::vector<std::vector<cv::Point>> SegPostProcessor::shiftAndScaleContours(
    const std::vector<std::vector<cv::Point>> &contours, double x_shift,
    double y_shift, double x_scale, double y_scale, double x_roi_shift,
    double y_roi_shift) const {
  std::vector<std::vector<cv::Point>> result;
  result.reserve(contours.size());

  for (const auto &c : contours) {
    std::vector<cv::Point> processed_contour;
    processed_contour.reserve(c.size());

    for (const auto &p : c) {
      int new_x = static_cast<int>((p.x + x_shift) * x_scale) + x_roi_shift;
      int new_y = static_cast<int>((p.y + y_shift) * y_scale) + y_roi_shift;
      processed_contour.emplace_back(new_x, new_y);
    }
    result.push_back(std::move(processed_contour));
  }
  return result;
}

bool SegPostProcessor::isLongitudinalSection(
    const std::vector<cv::Point> &contour) const {
  double aspect_ratio = getBoundingRectAspectRatio(contour);
  double eccentricity = getEccentricity(contour);
  return aspect_ratio >= longitudinal_section_aspect_ratio_ &&
         eccentricity > longitudinal_section_eccentricity_;
}

double SegPostProcessor::getBoundingRectAspectRatio(
    const std::vector<cv::Point> &contour) {
  cv::Rect rect = cv::boundingRect(contour);
  int long_side = std::max(rect.width, rect.height);
  int short_side = std::min(rect.width, rect.height);
  return static_cast<double>(long_side) / (short_side + 1e-6);
}

double
SegPostProcessor::getEccentricity(const std::vector<cv::Point> &contour) {
  if (contour.size() >= 5) {
    cv::RotatedRect ellipse = cv::fitEllipse(contour);
    int long_side = std::max(ellipse.size.height, ellipse.size.width);
    int short_side = std::min(ellipse.size.height, ellipse.size.width);
    return std::sqrt(
        1 - std::pow(static_cast<double>(short_side) / (long_side + 1e-6), 2));
  }
  return 0;
}

int SegPostProcessor::calculateFrameScanType(
    const std::vector<int> &scan_type_labels,
    const std::vector<double> &cc_areas) {
  if (cc_areas.empty()) {
    return 0;
  }
  double total_area = std::accumulate(cc_areas.begin(), cc_areas.end(), 0.0);
  double weighted_sum = 0;
  for (size_t i = 0; i < scan_type_labels.size(); ++i) {
    weighted_sum += scan_type_labels[i] * cc_areas[i];
  }
  return (weighted_sum / total_area) >= 0.5 ? 1 : 0;
}

std::vector<std::vector<cv::Point>> SegPostProcessor::processLumenContours(
    const cv::Mat &seg_crop, const cv::Rect &bbox, const SegmentResult &seg_ret,
    int seg_w, int seg_h) const {
  cv::Mat lumen_mask;
  cv::inRange(seg_crop, cv::Scalar(1), cv::Scalar(1), lumen_mask);
  auto [lumen_contours, _] = findAndFilterContours(
      lumen_mask, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, min_cc_area_ratio_);
  return shiftAndScaleContours(lumen_contours, bbox.x, bbox.y,
                               static_cast<double>(seg_ret.roi.width) / seg_w,
                               static_cast<double>(seg_ret.roi.height) / seg_h,
                               seg_ret.roi.x, seg_ret.roi.y);
}

std::vector<std::vector<cv::Point>> SegPostProcessor::processPlaqueContours(
    const cv::Mat &seg_crop, const cv::Rect &bbox, const SegmentResult &seg_ret,
    int seg_w, int seg_h) const {
  cv::Mat plaque_mask;
  cv::inRange(seg_crop, cv::Scalar(2), cv::Scalar(255), plaque_mask);
  auto [plaque_contours, _] = findAndFilterContours(
      plaque_mask, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE,
      min_plaque_area_ratio_, min_plaque_area_);
  return shiftAndScaleContours(plaque_contours, bbox.x, bbox.y,
                               static_cast<double>(seg_ret.roi.width) / seg_w,
                               static_cast<double>(seg_ret.roi.height) / seg_h,
                               seg_ret.roi.x, seg_ret.roi.y);
}
} // namespace cv_process