#ifndef __CV_PROCESS_PW_SAMPLER_HPP
#define __CV_PROCESS_PW_SAMPLER_HPP

#include "measure_types.hpp"

namespace cv_process {
class PWSampler {
public:
  PWSampler() = default;

  PWGateOutput getPWGate(const PWGateInput &input) const;

private:
  std::pair<std::vector<double>,
            std::vector<std::pair<cv::Point2f, cv::Point2f>>>
  computeShortestDistances(const Contour2f &contourA,
                           const Contour2f &contourB) const;

  Contour2f sortCurvePoints(const Contour2f &points) const;

  std::vector<cv::Point2f> findLineCurveIntersections(const Contour2f &curve,
                                                      const LineEquation &line,
                                                      int n_interp = 1000,
                                                      double tol = 1e-6) const;

  std::vector<ContourDiameterNormalInfo>
  getContourDiameterByCurveNormalDirection(const Contour2f &polyline,
                                           const Contour2f &contour_b,
                                           double step = 0.1,
                                           int window_size = 5,
                                           bool only_longest = true) const;

  MaxDiameterInfo getContourMaxDiameter(const Contour2f &points) const;

  ClosestPointsInfo getClosestPoints(const Contour2f &contourA,
                                     const Contour2f &contourB) const;

  Contour2f getContourPointsWithinDistance(const Contour2f &contourA,
                                           const cv::Point2f &P,
                                           double max_distance) const;

  std::optional<cv::Point2f>
  findIntersectionPoint(const Contour2f &contour,
                        const std::pair<cv::Point2f, cv::Point2f> &ray) const;

  // Uses KDTree
  std::pair<Contour2f, Contour2f>
  splitPointsByCurve(const Contour2f &curve, const Contour2f &points_to_split,
                     int k = 10) const;

  std::pair<Contour2f, Contour2f>
  splitPointsByLine(const Contour2f &points_on_line,
                    const cv::Point2f &pivot_point,
                    const LineEquation &line) const;

  // Line Math
  std::optional<LineEquation> getGeneralLine(
      const cv::Point2f &p1,
      const std::optional<cv::Point2f> &p2 = std::nullopt,
      const std::optional<cv::Point2f> &tangent_vector = std::nullopt) const;

  // tangent_vector is (dx, dy)
  std::optional<LineEquation>
  linePointSlopeToGeneral(const cv::Point2f &point,
                          const cv::Point2f &tangent_vector) const;

  LineEquation lineTwoPointsToGeneral(const cv::Point2f &point1,
                                      const cv::Point2f &point2) const;

  // Tangents and Normals
  std::optional<NearestTangentInfo>
  getNearestTangent(const Contour2f &polyline, const cv::Point2f &target_point,
                    int window_size = 5, int n_interp = 1000,
                    double eps = 1e-8) const; // Requires spline

  LineEquation getPerpendicular(const LineEquation &line,
                                const cv::Point2f &point) const;

  // Semantic helpers
  bool isContourUpperSideOfCurve(const Contour2f &contour_to_check,
                                 const Contour2f &reference_curve) const;

  LumenDiameterInfo
  getLumenDiameter(const std::vector<Contour2f> &vessel_contours,
                   const LineEquation &normal_line, const Contour2f &centerline,
                   const std::vector<Contour2f> &pleque_contours = {}) const;

  PWGate getDefaultPWGate(const cv::Rect &roi) const;

  // Mask and Contour Utilities (from measure_utils, blood_gate_utils, etc.)
  cv::Mat getMaxConnectedComponentSeg(const cv::Mat &mask) const;

  // Corresponds to Python's line_shifing
  LineEquation shiftLineEquation(const LineEquation &line,
                                 const cv::Point2f &move_xy,
                                 const cv::Point2f &resize_ratios_xy,
                                 bool move_first) const;

  // Corresponds to Python's calculate_points_distance_matrix
  // Returning a matrix of doubles. The size will be points.size() x
  // points.size().
  std::vector<std::vector<double>>
  calculatePointsDistanceMatrix(const Contour2f &points) const;

  // Utility functions from us_pipeline.carotis.modules that need to be part of
  // the class or utility namespace For get_coord_shift Returns: {move_xy,
  // scale_ratio_xy, move_first}
  std::tuple<cv::Point2f, cv::Point2f, bool>
  getCoordShift(const cv::Size &model_size, const cv::Rect &roi_xywh) const;

  // For seg_mask_to_contours or find_contours (simplified)
  // Python's find_contours has more options; this is a common usage.
  std::vector<Contour>
  findContoursFromMask(const cv::Mat &binary_mask,
                       int contour_mode = cv::RETR_EXTERNAL,
                       int contour_method = cv::CHAIN_APPROX_SIMPLE,
                       bool largest_only = false) const;

  std::pair<std::vector<Contour>, cv::Mat>
  findContours(const cv::Mat &mask, int contour_mode, int contour_method,
               bool only_largest) const;

  std::pair<std::vector<Contour>, std::vector<double>>
  getContoursAndFilterByArea(const cv::Mat &mask, int contour_mode,
                             int contour_method, bool only_largest,
                             float area_ratio_th, int area_th) const;

  ScanType getFrameScanType(
      const std::vector<Contour> &vessel_contours, // Original integer contours
      const std::vector<double> &vessel_contour_areas,
      float longitudinal_section_aspect_ratio_th,
      float longitudinal_section_eccentricity_th) const;

  // For contour_list_shifting
  std::vector<Contour2f> shiftContourList(const std::vector<Contour> &contours,
                                          const cv::Point2f &move_xy,
                                          const cv::Point2f &resize_ratios_xy,
                                          bool move_first) const;
  // Overload for Contour2f
  std::vector<Contour2f>
  shiftContourList(const std::vector<Contour2f> &contours,
                   const cv::Point2f &move_xy,
                   const cv::Point2f &resize_ratios_xy, bool move_first) const;

  // For get_centerline (this is a major dependency)
  Contour2f getCenterline(const cv::Mat &seg_mask, int centerline_label = -1,
                          bool order_points = true) const;

  // For get_line_degree (from blood_gate_utils)
  // Returns angle in degrees as per PWGate.blood_angle definition
  float getLineAngleDegrees(const LineEquation &line) const;

  Contour2f convertContourTo2f(const Contour &int_contour) const;

  std::vector<Contour2f>
  convertContourListTo2f(const std::vector<Contour> &int_contours) const;

  Contour convertContourTo2i(const Contour2f &float_contour) const;
};

} // namespace cv_process
#endif