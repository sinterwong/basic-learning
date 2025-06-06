#ifndef __CV_PROCESS_PW_LOCATOR_HPP
#define __CV_PROCESS_PW_LOCATOR_HPP

#include "vessel_types.hpp"

namespace cv_process {
class PWGateLocator {
public:
  /**
   * @brief 构造函数。
   * @param segMask 输入的单通道分割掩码 (CV_8UC1)。
   * @param roi 原始图像中的感兴趣区域 (ROI)。
   * @param plaqueLabel 掩码中代表斑块的像素值。
   * @param minCCAreaRatioTh 最小连通区域面积比例阈值。
   * @param longitudinalAspectRatioTh 判断为长轴切面的长宽比阈值。
   * @param longitudinalEccentricityTh 判断为长轴切面的离心率阈值。
   */
  PWGateLocator(const cv::Mat &segMask, const cv::Rect &roi,
                int plaqueLabel = 3, double minCCAreaRatioTh = 0.3,
                double longitudinalAspectRatioTh = 2.0,
                double longitudinalEccentricityTh = 0.5);

  /**
   * @brief 执行计算并返回最终的PW门参数。这是该类的主要入口点。
   * @return 计算出的PW门结果。
   */
  PWGate locate();

private:
  struct PointCompare {
    bool operator()(const cv::Point &a, const cv::Point &b) const {
      if (a.y < b.y)
        return true;
      if (a.y > b.y)
        return false;
      return a.x < b.x;
    }
  };

private:
  // --- 成员变量 (State) ---
  const cv::Mat &m_inputMask;
  const cv::Rect &m_roi;
  const int m_plaqueLabel;

  // 配置参数
  const double m_minCCAreaRatioTh;
  const double m_longitudinalAspectRatioTh;
  const double m_longitudinalEccentricityTh;

  // 内部状态
  cv::Mat m_processedMask;
  std::vector<Contour> m_vesselContours;
  std::vector<Contour> m_plaqueContours;
  Contourf m_vesselCenterline;
  ScanType m_scanType = ScanType::Others;

  // 坐标变换参数
  cv::Point2f m_coordShift{0.f, 0.f};
  cv::Point2f m_coordScale{1.f, 1.f};
  bool m_isMoveFirst = true;

  // --- 私有方法 (Behavior) ---

  // 1. 主流程控制
  void initialize();
  void classifyScanType();
  PWGate handleLongitudinal();
  PWGate handleCrossSectional();
  PWGate getDefaultPWGate() const;

  // 2. 坐标变换工具
  void calculateCoordTransform();
  cv::Point2f transformPoint(const cv::Point2f &point) const;
  PointList transformPoints(const PointList &points) const;
  LineEquation transformLine(const LineEquation &line) const;

  // 3. 几何计算辅助函数 (静态)
  static void getContoursAndFilterByArea(const cv::Mat &mask,
                                         std::vector<Contour> &contours,
                                         bool onlyLargest, double areaRatioTh);

  // 3. 几何计算辅助函数 (静态)
  // 3.1 中心线提取主函数
  static Contourf
  getCenterline(const cv::Mat &segMask, bool withSmooth, bool orderPoints,
                bool filterSegments,
                double minPathLengthRatioTh, // Changed from pixel to ratio
                double smoothingSigma);

  // 3.2 中心线提取的辅助函数
  static cv::Mat skeletonize(const cv::Mat &binaryImage);
  static Contourf orderSkeletonPoints(const cv::Mat &skeletonMask);
  static cv::Mat filterSkeletonSegments(const cv::Mat &skeleton,
                                        double minPathLengthRatioTh);
  static void findKeypoints(const cv::Mat &skeleton, cv::Mat &endpoints,
                            cv::Mat &junctions);

  static PointList findLineCurveIntersections(const Contourf &curve,
                                              const LineEquation &line,
                                              int n_interp = 1000,
                                              double tol = 1e-6);
  static std::optional<NormalIntersectionResult>
  getContourDiameterByCurveNormalDirection(const Contourf &polyline,
                                           const Contour &contour,
                                           bool onlyLongest = true,
                                           double step = 0.1,
                                           int window_size = 5);
  static void splitPointsByCurve(const Contourf &curve, const PointList &points,
                                 PointList &leftSide, PointList &rightSide,
                                 int k = 10);
  static std::optional<MeasurementCandidate>
  getLumenDiameter(const std::vector<Contour> &vesselContours,
                   const LineEquation &normalLine, const Contourf &centerline,
                   const std::vector<Contour> &plaqueContours);
  static bool isContourOnUpperSideOfCurve(const Contour &contour,
                                          const Contourf &curve);
  static std::tuple<double, cv::Point2f, cv::Point2f>
  getClosestPoints(const PointList &A, const PointList &B);
  static std::tuple<cv::Point2f, cv::Point2f, double>
  getContourMaxDiameter(const Contour &points);
  static PointList getContourPointsWithinDistance(const Contourf &curve,
                                                  const cv::Point2f &center,
                                                  double maxDistance);
  static std::optional<std::tuple<cv::Point2f, LineEquation, cv::Point2f>>
  getNearestTangent(const Contourf &polyline, const cv::Point2f &point,
                    int window_size = 5);

  // 4. 纯数学/几何辅助函数 (静态)
  static LineEquation getGeneralLine(const cv::Point2f &p1,
                                     const cv::Point2f &p2);
  static LineEquation
  getGeneralLineFromPointTangent(const cv::Point2f &p,
                                 const cv::Point2f &tangent);
  static LineEquation getPerpendicular(const LineEquation &line,
                                       const cv::Point2f &point);
  static double getLineAngle(const LineEquation &line);
  static double calculatePointsDistance(const cv::Point2f &p1,
                                        const cv::Point2f &p2);

  static Contourf convertContourTo2f(const Contour &int_contour);

  static Contour convertContourTo2i(const Contourf &float_contour);

  static std::vector<Contourf>
  convertContourListTo2f(const std::vector<Contour> &int_contours);
};
} // namespace cv_process
#endif