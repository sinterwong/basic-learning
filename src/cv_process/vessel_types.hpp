#ifndef __CV_PROCESS_VESSEL_TYPE_HPP_
#define __CV_PROCESS_VESSEL_TYPE_HPP_
#include <opencv2/opencv.hpp>
#include <utility>

namespace cv_process {

// 使用类型别名提高代码可读性
using Contour = std::vector<cv::Point>;
using Contourf = std::vector<cv::Point2f>;
using PointList = std::vector<cv::Point2f>;

// 替代 Python 的 ScanType Enum，使用强类型枚举
enum class ScanType { Longitudinal = 1, Others = 0 };

// 表示直线方程 ax + by + c = 0
struct LineEquation {
  double a = 0.0, b = 0.0, c = 0.0;
};

// 替代 Python 的 PWGate dataclass
struct PWGate {
  cv::Point2f center_point{0.f, 0.f}; // PW门中心点坐标
  double blood_angle = 0.0;           // 血流角度 [-180, 180]
  double lumen_diameter = -1.0;       // 管腔内径
  ScanType scan_type = ScanType::Others;
};

// 用于存储 getContourDiameterByCurveNormalDirection 的返回结果
struct NormalIntersectionResult {
  std::pair<cv::Point2f, cv::Point2f> intersection_points; // 最长线段的两个端点
  cv::Point2f curve_point;   // 对应的曲线上的一点
  cv::Point2f curve_tangent; // 该点处的切线向量
};

// 用于存储一次完整的管腔测量候选信息
struct MeasurementCandidate {
  double diameter;
  std::pair<cv::Point2f, cv::Point2f> diameter_points;
  LineEquation tangent_line;
  cv::Point2f gate_point;
};
} // namespace cv_process
#endif
