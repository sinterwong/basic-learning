#include "pw_locator.hpp"
#include "spline.h"
#include <stack>

namespace cv_process {

void thinningIteration(cv::Mat &im, int iter) {
  cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

  for (int i = 1; i < im.rows - 1; i++) {
    for (int j = 1; j < im.cols - 1; j++) {
      // 只处理前景点
      if (im.at<uchar>(i, j) == 0)
        continue;

      // 获取8邻域点的值 (0 或 1)
      uchar p2 = im.at<uchar>(i - 1, j) / 255;
      uchar p3 = im.at<uchar>(i - 1, j + 1) / 255;
      uchar p4 = im.at<uchar>(i, j + 1) / 255;
      uchar p5 = im.at<uchar>(i + 1, j + 1) / 255;
      uchar p6 = im.at<uchar>(i + 1, j) / 255;
      uchar p7 = im.at<uchar>(i + 1, j - 1) / 255;
      uchar p8 = im.at<uchar>(i, j - 1) / 255;
      uchar p9 = im.at<uchar>(i - 1, j - 1) / 255;

      // 条件1: 0->1 的模式转换次数
      int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
              (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
              (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
              (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);

      // 条件2: 邻域前景点数量
      int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

      // 条件3 和 4: 特定模式检查
      int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
      int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

      if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
        marker.at<uchar>(i, j) = 255; // 标记待删除点
      }
    }
  }
  // 从原图中减去被标记的点
  im &= ~marker;
}

PWGateLocator::PWGateLocator(const cv::Mat &segMask, const cv::Rect &roi,
                             int plaqueLabel, double minCCAreaRatioTh,
                             double longitudinalAspectRatioTh,
                             double longitudinalEccentricityTh)
    : m_inputMask(segMask), m_roi(roi), m_plaqueLabel(plaqueLabel),
      m_minCCAreaRatioTh(minCCAreaRatioTh),
      m_longitudinalAspectRatioTh(longitudinalAspectRatioTh),
      m_longitudinalEccentricityTh(longitudinalEccentricityTh) {
  if (m_inputMask.type() != CV_8UC1) {
    // 在实际项目中应抛出异常
    std::cerr << "Error: Input segMask must be CV_8UC1." << std::endl;
  }
}

// --- 公共主方法 ---
PWGate PWGateLocator::locate() {
  if (m_inputMask.empty() || m_inputMask.type() != CV_8UC1) {
    return getDefaultPWGate();
  }

  initialize();

  if (m_vesselContours.empty()) {
    return getDefaultPWGate();
  }

  classifyScanType();

  if (m_scanType == ScanType::Longitudinal) {
    return handleLongitudinal();
  } else {
    return handleCrossSectional();
  }
}

// --- 1. 主流程控制 ---
void PWGateLocator::initialize() {
  // 仅保留最大连通分量
  cv::Mat mask_max_cc;
  cv::connectedComponents(m_inputMask, mask_max_cc, 8, CV_32S);
  if (mask_max_cc.empty())
    return;

  double max_area = 0;
  int max_label = -1;
  std::map<int, double> areas;
  for (int r = 0; r < mask_max_cc.rows; ++r) {
    for (int c = 0; c < mask_max_cc.cols; ++c) {
      int label = mask_max_cc.at<int>(r, c);
      if (label > 0)
        areas[label]++;
    }
  }
  for (const auto &pair : areas) {
    if (pair.second > max_area) {
      max_area = pair.second;
      max_label = pair.first;
    }
  }
  if (max_label != -1) {
    m_processedMask = (mask_max_cc == max_label);
  } else {
    m_processedMask = cv::Mat::zeros(m_inputMask.size(), CV_8UC1);
  }

  // 提取血管和斑块轮廓
  getContoursAndFilterByArea(m_processedMask > 0, m_vesselContours, true,
                             m_minCCAreaRatioTh);
  getContoursAndFilterByArea(m_inputMask == m_plaqueLabel, m_plaqueContours,
                             false, m_minCCAreaRatioTh);

  calculateCoordTransform();
}

void PWGateLocator::classifyScanType() {
  m_scanType = ScanType::Others;
  if (m_vesselContours.empty())
    return;

  const auto &largest_contour = m_vesselContours[0];
  if (largest_contour.size() < 5)
    return; // minAreaRect需要至少5个点

  cv::RotatedRect box = cv::minAreaRect(largest_contour);
  float w = box.size.width;
  float h = box.size.height;
  if (w < 1e-6 || h < 1e-6)
    return;

  float aspect_ratio = std::max(w, h) / std::min(w, h);

  cv::Moments M = cv::moments(largest_contour);
  if (M.m00 == 0)
    return;
  double mu20 = M.m20 / M.m00 - (M.m10 / M.m00) * (M.m10 / M.m00);
  double mu02 = M.m02 / M.m00 - (M.m01 / M.m00) * (M.m01 / M.m00);
  double mu11 = M.m11 / M.m00 - (M.m10 / M.m00) * (M.m01 / M.m00);
  double term = std::sqrt(std::pow(mu20 - mu02, 2) + 4 * mu11 * mu11);
  double eccentricity = term / (mu20 + mu02);

  if (aspect_ratio > m_longitudinalAspectRatioTh &&
      eccentricity > m_longitudinalEccentricityTh) {
    m_scanType = ScanType::Longitudinal;
  }
}

PWGate PWGateLocator::handleLongitudinal() {
  m_vesselCenterline =
      getCenterline(m_processedMask, true, true, true, 0.05, 2.0);

  if (m_vesselCenterline.size() < 2)
    return getDefaultPWGate();

  std::vector<MeasurementCandidate> candidates;

  if (!m_plaqueContours.empty()) {
    for (const auto &plaque_contour : m_plaqueContours) {
      if (plaque_contour.size() < 3)
        continue;

      auto [p1, p2, plaque_max_diameter] =
          getContourMaxDiameter(plaque_contour);
      auto [dist, centerline_point, plaque_pt] = getClosestPoints(
          m_vesselCenterline, {plaque_contour.begin(), plaque_contour.end()});

      Contourf centerline_cut = getContourPointsWithinDistance(
          m_vesselCenterline, centerline_point, plaque_max_diameter);
      if (centerline_cut.size() < 2)
        continue;

      auto thickness_info = getContourDiameterByCurveNormalDirection(
          centerline_cut, plaque_contour, true, 1.0, 5);
      if (!thickness_info.has_value())
        continue;

      LineEquation tangent_line = getGeneralLineFromPointTangent(
          thickness_info->curve_point, thickness_info->curve_tangent);
      LineEquation normal_line =
          getPerpendicular(tangent_line, thickness_info->curve_point);

      auto lumen_info = getLumenDiameter(m_vesselContours, normal_line,
                                         m_vesselCenterline, m_plaqueContours);
      if (lumen_info.has_value()) {
        candidates.push_back(lumen_info.value());
      }
    }
  } else {
    cv::Rect vessel_bbox = cv::boundingRect(m_vesselContours[0]);
    cv::Point2f bbox_center(vessel_bbox.x + vessel_bbox.width / 2.0f,
                            vessel_bbox.y + vessel_bbox.height / 2.0f);

    auto tangent_info = getNearestTangent(m_vesselCenterline, bbox_center);
    if (tangent_info.has_value()) {
      auto [tangent_point, tangent_line, tangent_vec] = *tangent_info;
      LineEquation normal_line = getPerpendicular(tangent_line, tangent_point);
      auto lumen_info = getLumenDiameter(m_vesselContours, normal_line,
                                         m_vesselCenterline, {});
      if (lumen_info.has_value()) {
        candidates.push_back(lumen_info.value());
      }
    }
  }

  if (candidates.empty()) {
    return getDefaultPWGate();
  }

  // 找到直径最小的候选者
  std::sort(
      candidates.begin(), candidates.end(),
      [](const auto &a, const auto &b) { return a.diameter < b.diameter; });

  auto best_candidate = candidates[0];

  // 转换到ROI坐标系
  PointList diameter_points_roi =
      transformPoints({best_candidate.diameter_points.first,
                       best_candidate.diameter_points.second});
  cv::Point2f gate_point_roi = transformPoint(best_candidate.gate_point);
  LineEquation tangent_line_roi = transformLine(best_candidate.tangent_line);

  PWGate final_gate;
  final_gate.center_point = gate_point_roi;
  final_gate.lumen_diameter =
      calculatePointsDistance(diameter_points_roi[0], diameter_points_roi[1]);
  final_gate.blood_angle = getLineAngle(tangent_line_roi);
  final_gate.scan_type = ScanType::Longitudinal;

  return final_gate;
}

PWGate PWGateLocator::handleCrossSectional() {
  cv::Rect bbox = cv::boundingRect(m_vesselContours[0]);
  cv::Point2f center(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);

  PWGate gate;
  gate.center_point = transformPoint(center);
  gate.blood_angle = 0.0;
  gate.lumen_diameter = -1.0;
  gate.scan_type = ScanType::Others;
  return gate;
}

PWGate PWGateLocator::getDefaultPWGate() const {
  PWGate gate;
  gate.center_point = {m_roi.x + m_roi.width / 2.0f,
                       m_roi.y + m_roi.height / 2.0f};
  gate.scan_type = ScanType::Others;
  return gate;
}

// --- 2. 坐标变换工具 ---
void PWGateLocator::calculateCoordTransform() {
  cv::Size model_size = m_inputMask.size();
  float scale_x = static_cast<float>(m_roi.width) / model_size.width;
  float scale_y = static_cast<float>(m_roi.height) / model_size.height;

  m_coordScale = cv::Point2f(scale_x, scale_y);
  m_coordShift =
      cv::Point2f(static_cast<float>(m_roi.x), static_cast<float>(m_roi.y));
  m_isMoveFirst = false; // Corresponds to Python's get_coord_shift logic
}

cv::Point2f PWGateLocator::transformPoint(const cv::Point2f &point) const {
  if (m_isMoveFirst) {
    return cv::Point2f((point.x + m_coordShift.x) * m_coordScale.x,
                       (point.y + m_coordShift.y) * m_coordScale.y);
  } else {
    return cv::Point2f(point.x * m_coordScale.x + m_coordShift.x,
                       point.y * m_coordScale.y + m_coordShift.y);
  }
}

PointList PWGateLocator::transformPoints(const PointList &points) const {
  PointList transformed_points;
  transformed_points.reserve(points.size());
  for (const auto &p : points) {
    transformed_points.push_back(transformPoint(p));
  }
  return transformed_points;
}

LineEquation PWGateLocator::transformLine(const LineEquation &line) const {
  // This logic precisely matches the provided python `line_shifting`
  auto [a, b, c] = line;
  float m = m_coordShift.x, n = m_coordShift.y;
  float p = m_coordScale.x, q = m_coordScale.y;

  if (m_isMoveFirst) { // Python's `move_first = True`
    double c_prime = c - a * m - b * n;
    return {a * q, b * p, c_prime * p * q};
  } else { // Python's `move_first = False`
    double a_scaled = a / p;
    double b_scaled = b / q;
    return {a_scaled, b_scaled, c - a_scaled * m - b_scaled * n};
  }
}

// --- 3. 几何计算辅助函数 (静态实现) ---

void PWGateLocator::getContoursAndFilterByArea(const cv::Mat &mask,
                                               std::vector<Contour> &contours,
                                               bool onlyLargest,
                                               double areaRatioTh) {
  std::vector<Contour> all_contours;
  cv::findContours(mask, all_contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);

  if (all_contours.empty()) {
    contours.clear();
    return;
  }

  if (onlyLargest) {
    auto it = std::max_element(all_contours.begin(), all_contours.end(),
                               [](const auto &a, const auto &b) {
                                 return cv::contourArea(a) < cv::contourArea(b);
                               });
    contours = {*it};
    return;
  }

  std::vector<double> areas;
  double total_area = 0;
  for (const auto &c : all_contours) {
    double area = cv::contourArea(c);
    areas.push_back(area);
    total_area += area;
  }

  contours.clear();
  double area_th = total_area * areaRatioTh;
  for (size_t i = 0; i < all_contours.size(); ++i) {
    if (areas[i] >= area_th) {
      contours.push_back(all_contours[i]);
    }
  }
}

cv::Mat PWGateLocator::skeletonize(const cv::Mat &binaryImage) {
  // 确保输入是 CV_8UC1，值为 0 或 255
  cv::Mat img = binaryImage.clone();
  if (img.type() != CV_8UC1) {
    img.convertTo(img, CV_8U);
  }
  // 二值化，确保只有0和255
  cv::threshold(img, img, 127, 255, cv::THRESH_BINARY);

  cv::Mat prev = cv::Mat::zeros(img.size(), CV_8UC1);
  cv::Mat diff;

  do {
    thinningIteration(img, 0);
    thinningIteration(img, 1);
    cv::absdiff(img, prev, diff);
    img.copyTo(prev);
  }
  // 持续迭代直到图像不再变化
  while (cv::countNonZero(diff) > 0);

  return img;
}

Contourf PWGateLocator::orderSkeletonPoints(const cv::Mat &skeletonMask) {
  Contour points_yx;
  cv::findNonZero(skeletonMask, points_yx);

  if (points_yx.empty()) {
    return {};
  }

  // 1. 构建邻接表
  std::map<int, std::vector<int>> adj_list;
  std::map<cv::Point, int, decltype([](const cv::Point &a, const cv::Point &b) {
             return a.y < b.y || (a.y == b.y && a.x < b.x);
           })>
      point_to_idx;

  for (int i = 0; i < points_yx.size(); ++i) {
    point_to_idx[points_yx[i]] = i;
  }

  for (int i = 0; i < points_yx.size(); ++i) {
    const auto &p = points_yx[i];
    for (int dr = -1; dr <= 1; ++dr) {
      for (int dc = -1; dc <= 1; ++dc) {
        if (dr == 0 && dc == 0)
          continue;
        cv::Point neighbor(p.x + dc, p.y + dr);
        if (neighbor.x >= 0 && neighbor.x < skeletonMask.cols &&
            neighbor.y >= 0 && neighbor.y < skeletonMask.rows &&
            skeletonMask.at<uchar>(neighbor) > 0) {
          adj_list[i].push_back(point_to_idx[neighbor]);
        }
      }
    }
  }

  // 2. 找到端点（度为1的点）作为起点
  int start_node = -1;
  for (const auto &pair : adj_list) {
    if (pair.second.size() == 1) {
      start_node = pair.first;
      break;
    }
  }
  if (start_node == -1) { // 如果没有端点（例如，一个环），从0开始
    start_node = 0;
  }

  // 3. DFS遍历
  Contourf ordered_points;
  std::stack<int> s;
  std::vector<bool> visited(points_yx.size(), false);

  s.push(start_node);

  while (!s.empty()) {
    int u = s.top();
    s.pop();

    if (!visited[u]) {
      visited[u] = true;
      ordered_points.push_back(cv::Point2f(points_yx[u].x, points_yx[u].y));
      // 将邻居逆序入栈，以保持更自然的遍历顺序
      const auto &neighbors = adj_list[u];
      for (auto it = neighbors.rbegin(); it != neighbors.rend(); ++it) {
        if (!visited[*it]) {
          s.push(*it);
        }
      }
    }
  }

  return ordered_points;
}

cv::Mat PWGateLocator::filterSkeletonSegments(const cv::Mat &skeleton,
                                              double minPathLengthRatioTh) {

  if (cv::countNonZero(skeleton) == 0) {
    return skeleton.clone();
  }

  // 1. 找到所有关键点 (端点和交叉点)
  cv::Mat endpoints, junctions;
  findKeypoints(skeleton, endpoints, junctions);
  cv::Mat keypoints = endpoints | junctions;

  Contour keypoint_coords;
  cv::findNonZero(keypoints, keypoint_coords);

  // 如果关键点少于2个
  // (例如，一个孤立点或一个无端点的完美环)，则无法过滤，返回原图
  if (keypoint_coords.size() < 2) {
    return skeleton.clone();
  }

  // 2. 准备数据结构
  cv::Mat final_skeleton = cv::Mat::zeros(skeleton.size(), CV_8UC1);
  double min_path_length =
      static_cast<double>(cv::countNonZero(skeleton)) * minPathLengthRatioTh;

  // 用于存储发现的段: { {start_pt, end_pt}, path_points }
  std::vector<
      std::pair<std::pair<cv::Point, cv::Point>, std::vector<cv::Point>>>
      all_segments;

  // 用于防止重复处理同一段 (例如，A到B 和 B到A)
  std::set<std::pair<cv::Point, cv::Point>,
           decltype([](const auto &a, const auto &b) {
             PointCompare cmp;
             if (cmp(a.first, b.first))
               return true;
             if (cmp(b.first, a.first))
               return false;
             return cmp(a.second, b.second);
           })>
      processed_pairs;

  // 3. 从每个关键点出发，探索并记录所有段
  for (const auto &start_keypoint : keypoint_coords) {
    // 从当前关键点的每个邻居方向开始追踪路径
    for (int dr = -1; dr <= 1; ++dr) {
      for (int dc = -1; dc <= 1; ++dc) {
        if (dr == 0 && dc == 0)
          continue;

        cv::Point neighbor(start_keypoint.x + dc, start_keypoint.y + dr);

        // 确保邻居在图像内且在骨架上
        if (neighbor.x < 0 || neighbor.x >= skeleton.cols || neighbor.y < 0 ||
            neighbor.y >= skeleton.rows || skeleton.at<uchar>(neighbor) == 0) {
          continue;
        }

        // 创建规范化的关键点对以检查是否已处理
        cv::Point p1 = start_keypoint, p2 = neighbor;
        if (PointCompare()(p2, p1))
          std::swap(p1, p2);
        // 这里检查邻居是否也是关键点，如果是，则这是长度为1的段
        if (keypoints.at<uchar>(neighbor) > 0) {
          if (processed_pairs.find({p1, p2}) == processed_pairs.end()) {
            all_segments.push_back({{start_keypoint, neighbor}, {}});
            processed_pairs.insert({p1, p2});
          }
          continue; // 继续下一个邻居
        }

        // 开始追踪路径
        std::vector<cv::Point> current_path;
        cv::Point previous_point = start_keypoint;
        cv::Point current_point = neighbor;

        while (true) {
          // 检查当前点是否是关键点
          if (keypoints.at<uchar>(current_point) > 0) {
            cv::Point end_keypoint = current_point;
            cv::Point can_p1 = start_keypoint, can_p2 = end_keypoint;
            if (PointCompare()(can_p2, can_p1))
              std::swap(can_p1, can_p2);

            if (processed_pairs.find({can_p1, can_p2}) ==
                processed_pairs.end()) {
              all_segments.push_back(
                  {{start_keypoint, end_keypoint}, current_path});
              processed_pairs.insert({can_p1, can_p2});
            }
            break; // 路径追踪结束
          }

          current_path.push_back(current_point);

          // 寻找下一个点
          std::vector<cv::Point> next_points;
          for (int ndr = -1; ndr <= 1; ++ndr) {
            for (int ndc = -1; ndc <= 1; ++ndc) {
              if (ndr == 0 && ndc == 0)
                continue;
              cv::Point next_neighbor(current_point.x + ndc,
                                      current_point.y + ndr);
              if (next_neighbor == previous_point)
                continue; // 禁止回头

              if (next_neighbor.x >= 0 && next_neighbor.x < skeleton.cols &&
                  next_neighbor.y >= 0 && next_neighbor.y < skeleton.rows &&
                  skeleton.at<uchar>(next_neighbor) > 0) {
                next_points.push_back(next_neighbor);
              }
            }
          }

          // 在一个段的中间，应该只有一个前进方向
          if (next_points.size() != 1) {
            break; // 路径中断或遇到未标记的交叉点，停止追踪
          }

          previous_point = current_point;
          current_point = next_points[0];
        }
      }
    }
  }

  // 4. 根据规则过滤并绘制保留的段
  for (const auto &segment : all_segments) {
    const auto &start_p = segment.first.first;
    const auto &end_p = segment.first.second;
    const auto &path = segment.second;

    bool is_start_junction = junctions.at<uchar>(start_p) > 0;
    bool is_end_junction = junctions.at<uchar>(end_p) > 0;
    bool is_start_endpoint = endpoints.at<uchar>(start_p) > 0;
    bool is_end_endpoint = endpoints.at<uchar>(end_p) > 0;

    size_t path_length = path.size() + 2; // 路径点 + 两个端点

    bool keep_segment = false;
    // 规则1: 两端都是交叉点 -> 保留
    if (is_start_junction && is_end_junction) {
      keep_segment = true;
    }
    // 规则2: 两端都是端点 (构成一个独立的线段) -> 保留
    else if (is_start_endpoint && is_end_endpoint) {
      keep_segment = true;
    }
    // 规则3: 一端是交叉点，一端是端点 (一个分支) -> 检查长度
    else if ((is_start_junction && is_end_endpoint) ||
             (is_start_endpoint && is_end_junction)) {
      if (path_length >= min_path_length) {
        keep_segment = true;
      }
    }

    if (keep_segment) {
      // 绘制段
      final_skeleton.at<uchar>(start_p) = 255;
      final_skeleton.at<uchar>(end_p) = 255;
      for (const auto &p : path) {
        final_skeleton.at<uchar>(p) = 255;
      }
    }
  }

  return final_skeleton;
}

void PWGateLocator::findKeypoints(const cv::Mat &skeleton, cv::Mat &endpoints,
                                  cv::Mat &junctions) {
  if (skeleton.type() != CV_8UC1) {
    return;
  }

  cv::Mat skeleton_float;
  skeleton.convertTo(skeleton_float, CV_32F);

  skeleton_float /= 255.0;

  cv::Mat kernel = (cv::Mat_<float>(3, 3) << 1, 1, 1, 1, 0, 1, 1, 1, 1);

  cv::Mat neighbors;
  cv::filter2D(skeleton_float, neighbors, -1, kernel, cv::Point(-1, -1), 0,
               cv::BORDER_CONSTANT);

  endpoints = ((neighbors == 1) & (skeleton > 0));
  junctions = ((neighbors >= 3) & (skeleton > 0));
}

Contourf PWGateLocator::getCenterline(const cv::Mat &segMask, bool withSmooth,
                                      bool orderPoints, bool filterSegments,
                                      double minPathLengthRatioTh,
                                      double smoothingSigma) {
  // 1. 获取二值图像
  cv::Mat centerline_seg;
  if (segMask.type() != CV_8U) { // 假设输入segMask的值是标签
    centerline_seg = segMask > 0;
  } else {
    centerline_seg = segMask.clone();
  }

  // 2. 平滑
  if (withSmooth) {
    cv::Mat smoothed;
    cv::GaussianBlur(centerline_seg, smoothed, cv::Size(), smoothingSigma);
    centerline_seg = smoothed > 127; // 阈值处理
  }

  // 3. 骨架化
  cv::Mat centerline_mat = skeletonize(centerline_seg);

  cv::imwrite("centerline_mat.png", centerline_mat);

  // 4. 过滤
  if (filterSegments) {
    centerline_mat =
        filterSkeletonSegments(centerline_mat, minPathLengthRatioTh);

    cv::imwrite("centerline_filtered.png", centerline_mat);
  }

  // 5. 排序
  Contourf pt_list;
  if (orderPoints) {
    pt_list = orderSkeletonPoints(centerline_mat);
  } else {
    Contour points_yx;
    cv::findNonZero(centerline_mat, points_yx);
    for (const auto &p : points_yx) {
      pt_list.push_back(cv::Point2f(p.x, p.y));
    }
  }

  return pt_list;
}

std::optional<MeasurementCandidate> PWGateLocator::getLumenDiameter(
    const std::vector<Contour> &vesselContours, const LineEquation &normalLine,
    const Contourf &centerline, const std::vector<Contour> &plaqueContours) {
  // 1. 首先获取血管壁本身的上下边界点
  PointList vessel_boundary_pts;
  for (const auto &contour : vesselContours) {
    Contourf contour_f = convertContourTo2f(contour);
    auto intersections = findLineCurveIntersections(contour_f, normalLine);
    vessel_boundary_pts.insert(vessel_boundary_pts.end(), intersections.begin(),
                               intersections.end());
  }

  PointList vessel_upper_boundary, vessel_lower_boundary;
  splitPointsByCurve(centerline, vessel_boundary_pts, vessel_upper_boundary,
                     vessel_lower_boundary);

  // 2. 将血管壁边界作为最终边界的初始值
  PointList final_upper_boundary = vessel_upper_boundary;
  PointList final_lower_boundary = vessel_lower_boundary;

  // 3. 如果有斑块，用斑块边界替换血管壁边界
  if (!plaqueContours.empty()) {
    bool upper_replaced = false;
    bool lower_replaced = false;

    for (const auto &p_contour : plaqueContours) {
      if (p_contour.size() < 3)
        continue;

      Contourf p_contour_f = convertContourTo2f(p_contour);
      auto plaque_intersections =
          findLineCurveIntersections(p_contour_f, normalLine);

      if (plaque_intersections.empty()) {
        continue; // 此斑块与法线不相交
      }

      if (isContourOnUpperSideOfCurve(p_contour, centerline)) {
        // 如果是第一个上侧斑块，直接替换。
        // 如果已有上侧斑块，合并它们（理论上一个法线位置只有一个最窄点，但为鲁棒性可合并）
        if (!upper_replaced) {
          final_upper_boundary.clear();
          upper_replaced = true;
        }
        final_upper_boundary.insert(final_upper_boundary.end(),
                                    plaque_intersections.begin(),
                                    plaque_intersections.end());
      } else { // 斑块在下侧
        if (!lower_replaced) {
          final_lower_boundary.clear();
          lower_replaced = true;
        }
        final_lower_boundary.insert(final_lower_boundary.end(),
                                    plaque_intersections.begin(),
                                    plaque_intersections.end());
      }
    }
  }

  // 4. 检查最终的边界是否有效
  if (final_upper_boundary.empty() || final_lower_boundary.empty()) {
    return std::nullopt; // 如果任一边界没有点，则无法测量
  }

  // 5. 在最终确定的边界上计算最近点
  auto [diameter, p_upper, p_lower] =
      getClosestPoints(final_upper_boundary, final_lower_boundary);

  if (diameter < 0) { // getClosestPoints 失败
    return std::nullopt;
  }

  // --- 后续逻辑保持不变 ---
  MeasurementCandidate candidate;
  candidate.diameter = diameter;
  candidate.diameter_points = {p_upper, p_lower};
  candidate.gate_point = {(p_upper.x + p_lower.x) / 2.0f,
                          (p_upper.y + p_lower.y) / 2.0f};

  auto tangent_info = getNearestTangent(centerline, candidate.gate_point);
  if (tangent_info.has_value()) {
    candidate.tangent_line = std::get<1>(*tangent_info);
  } else {
    LineEquation diameter_line = getGeneralLine(p_upper, p_lower);
    candidate.tangent_line =
        getPerpendicular(diameter_line, candidate.gate_point);
  }

  return candidate;
}

bool PWGateLocator::isContourOnUpperSideOfCurve(const Contour &contour,
                                                const Contourf &curve) {
  Contourf contour_f = convertContourTo2f(contour);

  PointList upper, lower;
  splitPointsByCurve(curve, contour_f, upper, lower);
  return upper.size() > lower.size();
}

std::tuple<double, cv::Point2f, cv::Point2f>
PWGateLocator::getClosestPoints(const PointList &A, const PointList &B) {
  if (A.empty() || B.empty())
    return {-1, {}, {}};

  double min_dist_sq = std::numeric_limits<double>::max();
  cv::Point2f best_a, best_b;

  // Use FLANN for efficiency if point lists are large
  cv::Mat featuresA(A.size(), 2, CV_32F, const_cast<cv::Point2f *>(A.data()));
  cv::Mat featuresB(B.size(), 2, CV_32F, const_cast<cv::Point2f *>(B.data()));

  cv::flann::Index kdtree(featuresB, cv::flann::KDTreeIndexParams());

  for (size_t i = 0; i < A.size(); ++i) {
    std::vector<float> query = {A[i].x, A[i].y};
    std::vector<int> indices(1);
    std::vector<float> dists(1);
    kdtree.knnSearch(query, indices, dists, 1);

    if (dists[0] < min_dist_sq) {
      min_dist_sq = dists[0];
      best_a = A[i];
      best_b = B[indices[0]];
    }
  }
  return {std::sqrt(min_dist_sq), best_a, best_b};
}

PointList PWGateLocator::findLineCurveIntersections(const Contourf &curve,
                                                    const LineEquation &line,
                                                    int n_interp, double tol) {
  if (curve.size() < 2) {
    return {};
  }

  // 1. 准备数据并创建三次样条插值
  std::vector<double> t_orig(curve.size()), x_orig(curve.size()),
      y_orig(curve.size());
  for (size_t i = 0; i < curve.size(); ++i) {
    t_orig[i] = static_cast<double>(i);
    x_orig[i] = curve[i].x;
    y_orig[i] = curve[i].y;
  }

  tk::spline cs_x(t_orig, x_orig);
  tk::spline cs_y(t_orig, y_orig);

  // 2. 在密集采样点上计算直线方程的值
  std::vector<double> t_fine(n_interp);
  std::vector<double> f_values(n_interp);
  double t_max = static_cast<double>(curve.size() - 1);

  for (int i = 0; i < n_interp; ++i) {
    t_fine[i] = i * t_max / (n_interp - 1);
    double x = cs_x(t_fine[i]);
    double y = cs_y(t_fine[i]);
    f_values[i] = line.a * x + line.b * y + line.c;
  }

  // 3. 查找符号变化，定位交点区间
  PointList intersections;
  for (int i = 0; i < n_interp - 1; ++i) {
    if (f_values[i] * f_values[i + 1] < 0) { // 符号发生变化，说明存在根
      // 4. 使用割线法 (Secant Method) 在区间 [t_fine[i], t_fine[i+1]]
      // 内精确求解
      double t0 = t_fine[i];
      double t1 = t_fine[i + 1];
      double f0 = f_values[i];
      double f1 = f_values[i + 1];

      for (int iter = 0; iter < 10; ++iter) { // 10次迭代通常足够
        if (std::abs(f1 - f0) < 1e-9)
          break; // 避免除以零

        double t_next = t1 - f1 * (t1 - t0) / (f1 - f0);

        t0 = t1;
        f0 = f1;
        t1 = t_next;
        f1 = line.a * cs_x(t1) + line.b * cs_y(t1) + line.c;

        if (std::abs(f1) < tol)
          break; // 收敛
      }

      // 检查最终解的有效性
      if (std::abs(f1) < tol) {
        intersections.push_back({(float)cs_x(t1), (float)cs_y(t1)});
      }
    }
  }

  // 5. 去除重复的交点
  if (intersections.size() > 1) {
    std::sort(intersections.begin(), intersections.end(),
              [](const cv::Point2f &a, const cv::Point2f &b) {
                return a.x < b.x || (a.x == b.x && a.y < b.y);
              });

    auto last = std::unique(intersections.begin(), intersections.end(),
                            [tol](const cv::Point2f &a, const cv::Point2f &b) {
                              return cv::norm(a - b) < tol;
                            });

    intersections.erase(last, intersections.end());
  }

  return intersections;
}

std::optional<NormalIntersectionResult>
PWGateLocator::getContourDiameterByCurveNormalDirection(
    const Contourf &polyline, const Contour &contour, bool onlyLongest,
    double step, int window_size) {
  if (polyline.size() < 2 || contour.size() < 3) {
    return std::nullopt;
  }

  // 1. 创建三次样条插值
  std::vector<double> t_orig(polyline.size()), x_orig(polyline.size()),
      y_orig(polyline.size());
  for (size_t i = 0; i < polyline.size(); ++i) {
    t_orig[i] = static_cast<double>(i);
    x_orig[i] = polyline[i].x;
    y_orig[i] = polyline[i].y;
  }
  tk::spline cs_x(t_orig, x_orig, tk::spline::cspline);
  tk::spline cs_y(t_orig, y_orig, tk::spline::cspline);

  // 2. 生成密集采样点
  double t_max = polyline.size() - 1.0;
  int num_samples = static_cast<int>(t_max / step) + 1;
  std::vector<cv::Point2f> sampled_points(num_samples);
  std::vector<double> t_fine(num_samples);
  for (int i = 0; i < num_samples; ++i) {
    t_fine[i] = std::min(i * step, t_max);
    sampled_points[i] = cv::Point2f(cs_x(t_fine[i]), cs_y(t_fine[i]));
  }

  std::vector<NormalIntersectionResult> all_results;
  const double epsilon = 1e-9;

  // 3. 遍历每个采样点
  for (int i = 0; i < num_samples; ++i) {
    // 3a. 计算法线方向 (使用邻域)
    double t_start = std::max(0.0, t_fine[i] - window_size * step);
    double t_end = std::min(t_max, t_fine[i] + window_size * step);
    if (std::abs(t_end - t_start) < epsilon)
      continue;

    cv::Point2f p_start(cs_x(t_start), cs_y(t_start));
    cv::Point2f p_end(cs_x(t_end), cs_y(t_end));
    cv::Point2f tangent = p_end - p_start;
    double norm = cv::norm(tangent);
    if (norm < epsilon)
      continue;
    tangent /= norm;

    cv::Point2f normal(-tangent.y, tangent.x); // 左法线

    // 3b. 计算法线与轮廓所有边的交点
    std::vector<double> intersection_distances; // 存储交点沿法线方向的距离 t
    cv::Point2f ray_origin = sampled_points[i];

    for (size_t j = 0; j < contour.size(); ++j) {
      cv::Point2f p1 = contour[j];
      cv::Point2f p2 = contour[(j + 1) % contour.size()];
      cv::Point2f edge_vec = p2 - p1;

      // 求解方程: ray_origin + t * normal = p1 + s * edge_vec
      // 使用克拉默法则求解 t 和 s
      double det = normal.x * edge_vec.y - normal.y * edge_vec.x;
      if (std::abs(det) < epsilon)
        continue; // 平行线

      double t = ((p1.x - ray_origin.x) * edge_vec.y -
                  (p1.y - ray_origin.y) * edge_vec.x) /
                 det;
      double s = (normal.x * (p1.y - ray_origin.y) -
                  normal.y * (p1.x - ray_origin.x)) /
                 -det;

      if (s >= 0 && s <= 1) { // 交点在轮廓边上
        intersection_distances.push_back(t);
      }
    }

    if (intersection_distances.size() < 2)
      continue;

    // 3c. 排序并筛选内部线段
    std::sort(intersection_distances.begin(), intersection_distances.end());

    double max_len = -1.0;
    double best_t1 = 0, best_t2 = 0;

    for (size_t j = 0; j < intersection_distances.size() - 1; ++j) {
      double t1 = intersection_distances[j];
      double t2 = intersection_distances[j + 1];

      // 检查中点是否在轮廓内
      double mid_t = (t1 + t2) / 2.0;
      cv::Point2f mid_point = ray_origin + mid_t * normal;

      if (cv::pointPolygonTest(contour, mid_point, false) >= 0) {
        double len = std::abs(t2 - t1);
        if (len > max_len) {
          max_len = len;
          best_t1 = t1;
          best_t2 = t2;
        }
      }
    }

    if (max_len > 0) {
      NormalIntersectionResult res;
      res.intersection_points.first = ray_origin + best_t1 * normal;
      res.intersection_points.second = ray_origin + best_t2 * normal;
      res.curve_point = sampled_points[i];
      res.curve_tangent = tangent;
      all_results.push_back(res);
    }
  }

  if (all_results.empty()) {
    return std::nullopt;
  }

  // 4. 找到最长的线段
  auto it = std::max_element(
      all_results.begin(), all_results.end(),
      [](const NormalIntersectionResult &a, const NormalIntersectionResult &b) {
        return cv::norm(a.intersection_points.first -
                        a.intersection_points.second) <
               cv::norm(b.intersection_points.first -
                        b.intersection_points.second);
      });

  return *it;
}

// --- 4. Pure math/geometry helpers ---
double PWGateLocator::getLineAngle(const LineEquation &line) {
  // Angle of the line's direction vector (b, -a) with the positive x-axis.
  double angle_rad = std::atan2(-line.a, line.b);
  double angle_deg = angle_rad * 180.0 / CV_PI;
  // The problem defines clockwise as positive, which is opposite to standard
  // math. And from x-axis. So we might need to adjust. Python's get_line_degree
  // might have specific conventions. Let's assume standard for now. The python
  // code returns an angle where顺时针为正，so we negate it.
  return -angle_deg;
}

double PWGateLocator::calculatePointsDistance(const cv::Point2f &p1,
                                              const cv::Point2f &p2) {
  return cv::norm(p1 - p2);
}

Contourf PWGateLocator::convertContourTo2f(const Contour &int_contour) {
  Contourf float_contour;
  float_contour.reserve(int_contour.size());
  for (const auto &pt : int_contour) {
    float_contour.emplace_back(static_cast<float>(pt.x),
                               static_cast<float>(pt.y));
  }
  return float_contour;
}

Contour PWGateLocator::convertContourTo2i(const Contourf &float_contour) {
  Contour int_contour;
  int_contour.reserve(float_contour.size());
  for (const auto &pt : float_contour) {
    int_contour.emplace_back(cv::saturate_cast<int>(pt.x),
                             cv::saturate_cast<int>(pt.y));
  }
  return int_contour;
}

std::vector<Contourf> PWGateLocator::convertContourListTo2f(
    const std::vector<Contour> &int_contours) {
  std::vector<Contourf> float_contours;
  float_contours.reserve(int_contours.size());
  for (const auto &c : int_contours) {
    float_contours.push_back(convertContourTo2f(c));
  }
  return float_contours;
}

void PWGateLocator::splitPointsByCurve(const Contourf &curve,
                                       const PointList &points,
                                       PointList &leftSide,
                                       PointList &rightSide, int k) {
  leftSide.clear();
  rightSide.clear();

  if (curve.size() < 2 || points.empty()) {
    return;
  }

  // 1. 为曲线上的每个点计算局部法线
  std::vector<cv::Point2f> normals(curve.size());
  for (size_t i = 0; i < curve.size(); ++i) {
    int start_idx = std::max(0, static_cast<int>(i) - k / 2);
    int end_idx = std::min(static_cast<int>(curve.size() - 1),
                           static_cast<int>(i) + k / 2);

    cv::Point2f tangent_vec = curve[end_idx] - curve[start_idx];
    double norm = cv::norm(tangent_vec);
    if (norm > 1e-6) {
      tangent_vec /= norm;
    }
    normals[i] = cv::Point2f(-tangent_vec.y, tangent_vec.x); // 左法线
  }

  // 2. 为所有待分类点建立 KD-Tree 以快速查找最近的曲线点
  cv::Mat curve_mat(curve.size(), 2, CV_32F,
                    const_cast<cv::Point2f *>(curve.data()));
  cv::flann::Index kdtree(curve_mat, cv::flann::KDTreeIndexParams());

  // 3. 对每个点进行分类
  for (const auto &p : points) {
    std::vector<float> query = {p.x, p.y};
    std::vector<int> indices(1);
    std::vector<float> dists(1);
    kdtree.knnSearch(query, indices, dists, 1);

    int nearest_idx = indices[0];

    // 4. 计算位置向量并判断左右侧
    cv::Point2f vector_to_curve = p - curve[nearest_idx];
    double dot_product = vector_to_curve.dot(normals[nearest_idx]);

    if (dot_product > 0) {
      leftSide.push_back(p);
    } else {
      rightSide.push_back(p);
    }
  }
}

std::tuple<cv::Point2f, cv::Point2f, double>
PWGateLocator::getContourMaxDiameter(const Contour &points) {
  if (points.size() < 2) {
    return {{}, {}, -1.0};
  }

  double max_dist_sq = -1.0;
  cv::Point2f p1, p2;

  for (size_t i = 0; i < points.size(); ++i) {
    for (size_t j = i + 1; j < points.size(); ++j) {
      // double dist_sq = cv::normL2Sqr(points[i] - points[j]);
      cv::Point2f diff = points[i] - points[j];
      double dist_sq = diff.dot(diff);
      if (dist_sq > max_dist_sq) {
        max_dist_sq = dist_sq;
        p1 = points[i];
        p2 = points[j];
      }
    }
  }
  return {p1, p2, std::sqrt(max_dist_sq)};
}

PointList PWGateLocator::getContourPointsWithinDistance(
    const Contourf &curve, const cv::Point2f &center, double maxDistance) {
  PointList result_points;
  if (curve.empty()) {
    return result_points;
  }

  double max_dist_sq = maxDistance * maxDistance;

  for (const auto &point : curve) {
    cv::Point2f diff = point - center;
    if (diff.dot(diff) <= max_dist_sq) {
      result_points.push_back(point);
    }
  }
  return result_points;
}

std::optional<std::tuple<cv::Point2f, LineEquation, cv::Point2f>>
PWGateLocator::getNearestTangent(const Contourf &polyline,
                                 const cv::Point2f &point, int window_size) {
  if (polyline.size() < 2) {
    return std::nullopt;
  }

  // 1. 在曲线上找到离目标点最近的点
  double min_dist_sq = std::numeric_limits<double>::max();
  int nearest_idx = -1;
  for (size_t i = 0; i < polyline.size(); ++i) {
    // double dist_sq = cv::normL2Sqr(polyline[i] - point);
    cv::Point2f diff = polyline[i] - point;
    double dist_sq = diff.dot(diff);
    if (dist_sq < min_dist_sq) {
      min_dist_sq = dist_sq;
      nearest_idx = i;
    }
  }

  cv::Point2f base_point = polyline[nearest_idx];

  // 2. 计算该最近点处的切线
  int start_idx = std::max(0, nearest_idx - window_size);
  int end_idx = std::min(static_cast<int>(polyline.size() - 1),
                         nearest_idx + window_size);

  cv::Point2f tangent_vec = polyline[end_idx] - polyline[start_idx];
  double norm = cv::norm(tangent_vec);
  if (norm < 1e-6) { // 如果切线向量太小，无法确定方向
    if (polyline.size() > 1) {
      tangent_vec = polyline[1] - polyline[0]; // 使用全局方向作为备用
      norm = cv::norm(tangent_vec);
      if (norm < 1e-6)
        return std::nullopt;
    } else {
      return std::nullopt;
    }
  }
  tangent_vec /= norm;

  // 3. 构造切线方程
  LineEquation tangent_line =
      getGeneralLineFromPointTangent(base_point, tangent_vec);

  return std::make_tuple(base_point, tangent_line, tangent_vec);
}

// --- 4. 纯数学/几何辅助函数 (静态实现) [补充] ---

LineEquation PWGateLocator::getGeneralLine(const cv::Point2f &p1,
                                           const cv::Point2f &p2) {
  double x1 = p1.x, y1 = p1.y;
  double x2 = p2.x, y2 = p2.y;

  double a = y2 - y1;
  double b = x1 - x2;
  double c = -a * x1 - b * y1; // or x2 * y1 - x1 * y2

  return {a, b, c};
}

LineEquation
PWGateLocator::getGeneralLineFromPointTangent(const cv::Point2f &p,
                                              const cv::Point2f &tangent) {
  // 法线向量是 (-tangent.y, tangent.x)
  double a = -tangent.y;
  double b = tangent.x;
  double c = -a * p.x - b * p.y;

  return {a, b, c};
}

LineEquation PWGateLocator::getPerpendicular(const LineEquation &line,
                                             const cv::Point2f &point) {
  // 原直线的方向向量是 (line.b, -line.a)
  // 这也是新直线的法线向量
  double a_new = line.b;
  double b_new = -line.a;
  double c_new = -a_new * point.x - b_new * point.y;

  return {a_new, b_new, c_new};
}
} // namespace cv_process
