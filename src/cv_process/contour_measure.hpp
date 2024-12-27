#ifndef __CAROTIS_MEASURE_HPP_
#define __CAROTIS_MEASURE_HPP_

#include "measure_types.hpp"

namespace cv_process {
class ContourMeasure {
public:
  ContourMeasure(float dist_thre = 5, float angle_weight = 0.4)
      : dist_thre(dist_thre), angle_weight(angle_weight) {}

  std::pair<Contour, Contour> process(const MeasureInput &seg_ret,
                                      MeasureRet &measurement) const;

private:
  double calculateDistance(cv::Point p1, cv::Point p2) const;
  double dotProduct(cv::Point2f v1, cv::Point2f v2) const;
  double calculateExtension(const std::vector<cv::Point> &plaque_contour) const;
  cv::Point2f calculateIntersection(cv::Point2f p1, cv::Point2f p2,
                                    cv::Point2f p3, cv::Point2f p4) const;
  std::pair<Contour, Contour> calcThicknessAndLength(
      std::vector<std::pair<Contour, Contour>> const &contour_pairs,
      MeasureRet &measurement) const;

private:
  float dist_thre;
  float angle_weight;
};

} // namespace cv_process
#endif
