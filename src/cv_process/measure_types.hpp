#ifndef __CV_PROCESS_MEASURE_TYPE_HPP_
#define __CV_PROCESS_MEASURE_TYPE_HPP_
#include <opencv2/opencv.hpp>
#include <utility>

namespace cv_process {

using Contour = std::vector<cv::Point>;

// For precise geometric calculations
using Contour2f = std::vector<cv::Point2f>;

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

// Corresponds to Python's ScanType enum
enum class ScanType {
  Longitudinal = 1,
  Others = 0 // Includes Cross-sectional
};

// Corresponds to Python's PWGate dataclass
struct PWGate {
  cv::Point2f center_point; // (x, y)
  float blood_angle; // Degrees, [-180, 180], clockwise positive from x-axis
  float lumen_diameter;
  ScanType scan_type;
};

// For representing a line in general form: ax + by + c = 0
struct LineEquation {
  double a, b, c;
};

// For results from get_contour_diameter_by_curve_normal_direction
struct ContourDiameterNormalInfo {
  std::pair<cv::Point2f, cv::Point2f>
      segment_endpoints;               // The diameter/thickness segment
  cv::Point2f curve_point_on_polyline; // Point on the polyline (centerline)
  cv::Point2f tangent_at_curve_point;  // Tangent vector at that point
};

// For results from get_contour_max_diameter
struct MaxDiameterInfo {
  cv::Point2f point1;
  cv::Point2f point2;
  double distance;
};

// For results from get_closest_points
struct ClosestPointsInfo {
  double distance;
  cv::Point2f point_on_A;
  cv::Point2f point_on_B;
};

// For results from get_nearest_tangent
struct NearestTangentInfo {
  cv::Point2f base_point_on_curve;
  LineEquation tangent_line_equation;
  cv::Point2f unit_tangent_vector;
};

// For results from get_lumen_diameter function
struct LumenDiameterInfo {
  double diameter = -1.0; // Default to invalid
  std::pair<cv::Point2f, cv::Point2f> diameter_endpoints;

  bool isValid() const { return diameter >= 0.0; }
};

// Input structure for get_pw_gate
struct PWGateInput {
  cv::Mat seg_mask;
  cv::Rect roi;
  float min_cc_area_ratio_th = 0.3f;
  float min_cc_area_th = 0.f;
  float longitudinal_section_aspect_ratio_th = 2.0f;
  float longitudinal_section_eccentricity_th = 0.5f;
  int pleque_label = 3; // Default in python, test_pw_gate uses 4
  bool debug = false;
};

// Output structure for get_pw_gate (if debug info is needed)
struct PWGateOutput {
  PWGate gate;
  struct DebugLumenInfo {
    double diameter;
    std::pair<cv::Point2f, cv::Point2f> diameter_points; // lumen_diamter_pts
    LineEquation tangent_line;                           // vessel_tangent_line
    cv::Point2f tangent_vector; // vessel_tangent (vector)
  };
  std::vector<DebugLumenInfo> debug_lumen_infos;
};

} // namespace cv_process
#endif
