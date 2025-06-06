#include "pw_sampler.hpp"
#include <numeric>
namespace cv_process {

static double cross_product(const cv::Point2f &v1, const cv::Point2f &v2) {
  return v1.x * v2.y - v1.y * v2.x;
}

PWGateOutput PWSampler::getPWGate(const PWGateInput &input) const {
  PWGateOutput output;
  // output.debug_lumen_infos will be filled if input.debug is true

  // Python roi is YXYX, input.roi is cv::Rect (x,y,w,h)
  // seg_mask is model output size.
  // 1. Initialization and Preprocessing
  auto [roi_move_xy, roi_scale_ratio_xy, move_first] = getCoordShift(
      cv::Size(input.seg_mask.cols, input.seg_mask.rows), input.roi);

  cv::Mat current_seg_mask = getMaxConnectedComponentSeg(input.seg_mask);

  if (current_seg_mask.empty()) {
    std::cerr << "Warning: getMaxConnectedComponentSeg returned empty mask."
              << std::endl;
    output.gate = getDefaultPWGate(input.roi);
    return output;
  }

  auto [vessel_contours_i, vessel_contour_areas] = getContoursAndFilterByArea(
      current_seg_mask > 0, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, true,
      input.min_cc_area_ratio_th, input.min_cc_area_th);

  // Convert integer contours to float contours (Contour2f)
  std::vector<Contour2f> vessel_contours =
      convertContourListTo2f(vessel_contours_i);

  if (vessel_contours.empty()) {
    std::cerr << "Warning: getContoursAndFilterByArea returned empty contours."
              << std::endl;
    output.gate = getDefaultPWGate(input.roi);
    return output;
  }

  // 2. Determine Scan Type
  ScanType frame_scan_type =
      getFrameScanType(vessel_contours_i, vessel_contour_areas,
                       input.longitudinal_section_aspect_ratio_th,
                       input.longitudinal_section_eccentricity_th);

  // 3. Process based on ScanType
  if (frame_scan_type == ScanType::Others) {
    // Transform vessel contour to original ROI coordinates
    std::vector<Contour2f> shifted_vessel_contours = shiftContourList(
        vessel_contours, roi_move_xy, roi_scale_ratio_xy, move_first);

    if (shifted_vessel_contours.empty()) {
      output.gate = getDefaultPWGate(input.roi);
      return output;
    }

    // Use boundingRect on the shifted contour (which is now Contour2f)
    // cv::boundingRect needs vector<Point>
    Contour shifted_vessel_contour_i =
        convertContourTo2i(shifted_vessel_contours[0]);
    cv::Rect vessel_bbox = cv::boundingRect(shifted_vessel_contour_i);

    cv::Point2f center_pt(
        static_cast<float>(vessel_bbox.x) + vessel_bbox.width / 2.0f,
        static_cast<float>(vessel_bbox.y) + vessel_bbox.height / 2.0f);
    output.gate = {center_pt, 0.0f, -1.0f, ScanType::Others};
    return output;

  } else { // ScanType::Longitudinal
    // Extract pleque contours (on model size mask)
    cv::Mat pleque_mask = (current_seg_mask == input.pleque_label);

    cv::imwrite("vis_pleque_mask.png", pleque_mask * 255);
    auto [pleque_contours_i, pleque_contour_areas] = getContoursAndFilterByArea(
        pleque_mask, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, false,
        input.min_cc_area_ratio_th, input.min_cc_area_th);

    std::vector<Contour2f> pleque_contours =
        convertContourListTo2f(pleque_contours_i);

    // Get centerline (on model size mask)
    // centerline_label=-1 means any positive value in seg_mask after getMaxCC
    // order_points=True is important
    Contour2f vessel_centerline_model =
        getCenterline(current_seg_mask, -1, true);
    // Python `get_centerline` can return empty if it fails.
    if (vessel_centerline_model.empty()) {
      std::cerr
          << "Warning: Vessel centerline extraction failed or returned empty."
          << std::endl;
      output.gate = getDefaultPWGate(input.roi); // Fallback
      return output;
    }
    // The Python code had a commented out sort_curve_points here. Assuming
    // getCenterline handles ordering.
    // Store results in model coordinates first
    std::vector<PWGateOutput::DebugLumenInfo> calculated_lumen_infos_model;

    if (!pleque_contours.empty()) {
      for (const auto &pleque_contour_model : pleque_contours) {
        if (pleque_contour_model.size() < 2)
          continue;

        MaxDiameterInfo mdi = getContourMaxDiameter(pleque_contour_model);
        ClosestPointsInfo cpi_center_pleque =
            getClosestPoints(vessel_centerline_model, pleque_contour_model);
        cv::Point2f centerline_point_near_pleque = cpi_center_pleque.point_on_A;

        Contour2f centerline_cut_model = getContourPointsWithinDistance(
            vessel_centerline_model, centerline_point_near_pleque,
            mdi.distance);
        if (centerline_cut_model.empty())
          continue;

        // get_contour_diameter_by_curve_normal_direction IS SKIPPED
        // This is a critical part for plaque-based measurement.
        // For now, we cannot proceed with this path accurately.
        // We would need its output: thickness_points, vessel_tanget_point,
        // vessel_tangent
        std::cerr
            << "Warning: Longitudinal PWGate with plaque relies on "
               "getContourDiameterByCurveNormalDirection, which is skipped."
            << std::endl;
        // Let's try to put a placeholder logic if possible or skip this loop
        // iteration. As a placeholder: try to find a point on
        // centerline_cut_model, and a pseudo-tangent. This is highly
        // speculative and likely incorrect without the proper function.
        if (centerline_cut_model.size() >= 2) {
          cv::Point2f vessel_tangent_point_model =
              centerline_cut_model[centerline_cut_model.size() / 2];
          cv::Point2f vessel_tangent_model =
              centerline_cut_model.back() - centerline_cut_model.front();
          float norm_tangent = cv::norm(vessel_tangent_model);
          if (norm_tangent > 1e-6)
            vessel_tangent_model /= norm_tangent;
          else
            vessel_tangent_model = cv::Point2f(1, 0); // Default tangent

          auto tangent_line_opt = linePointSlopeToGeneral(
              vessel_tangent_point_model, vessel_tangent_model);
          if (!tangent_line_opt)
            continue;
          LineEquation vessel_tangent_line_model = *tangent_line_opt;
          LineEquation vessel_normal_line_model = getPerpendicular(
              vessel_tangent_line_model, vessel_tangent_point_model);

          LumenDiameterInfo ldi_model =
              getLumenDiameter(vessel_contours, vessel_normal_line_model,
                               vessel_centerline_model, pleque_contours);
          if (!ldi_model.isValid())
            continue;

          // PW gate point is mean of lumen diameter points
          cv::Point2f pw_gate_point_model =
              (ldi_model.diameter_endpoints.first +
               ldi_model.diameter_endpoints.second) *
              0.5f;
          // Recalculate tangent line at this new pw_gate_point_model
          auto final_tangent_line_opt = linePointSlopeToGeneral(
              pw_gate_point_model, vessel_tangent_model);
          if (!final_tangent_line_opt)
            continue;

          calculated_lumen_infos_model.push_back(
              {ldi_model.diameter, ldi_model.diameter_endpoints,
               *final_tangent_line_opt, vessel_tangent_model});
        }
        // End of placeholder for skipped
        // getContourDiameterByCurveNormalDirection
      }

      if (!calculated_lumen_infos_model.empty()) {
        std::sort(calculated_lumen_infos_model.begin(),
                  calculated_lumen_infos_model.end(),
                  [](const PWGateOutput::DebugLumenInfo &a,
                     const PWGateOutput::DebugLumenInfo &b) {
                    return a.diameter < b.diameter; // Smallest diameter first
                  });
      }

    } else { // No plaques
      // get_nearest_tangent IS SKIPPED
      // This is critical for no-plaque case.
      std::cerr << "Warning: Longitudinal PWGate without plaque relies on "
                   "getNearestTangent, which is skipped."
                << std::endl;
      // Placeholder logic: use center of bounding box of vessel contour, find
      // nearest point on centerline
      if (vessel_contours_i.empty() ||
          vessel_centerline_model.empty()) { // Should not happen if we are here
        output.gate = getDefaultPWGate(input.roi);
        return output;
      }
      cv::Rect vessel_bbox_model =
          cv::boundingRect(vessel_contours_i[0]); // on model coords
      cv::Point2f vessel_bbox_center_model(
          static_cast<float>(vessel_bbox_model.x) +
              vessel_bbox_model.width / 2.0f,
          static_cast<float>(vessel_bbox_model.y) +
              vessel_bbox_model.height / 2.0f);

      // Find point on centerline closest to bbox_center
      ClosestPointsInfo cpi_center_bbox =
          getClosestPoints(vessel_centerline_model, {vessel_bbox_center_model});
      cv::Point2f vessel_tangent_point_model =
          cpi_center_bbox.point_on_A; // Point on centerline

      // Estimate tangent at this point (simple diff if centerline is ordered)
      cv::Point2f vessel_tangent_model;
      // Find index of vessel_tangent_point_model on vessel_centerline_model
      int N = vessel_centerline_model.size();
      int idx_on_cl = -1;
      for (int k = 0; k < N; ++k) {
        if (cv::norm(vessel_centerline_model[k] - vessel_tangent_point_model) <
            1e-3) {
          idx_on_cl = k;
          break;
        }
      }
      if (idx_on_cl != -1 && N >= 2) {
        if (idx_on_cl == 0)
          vessel_tangent_model =
              vessel_centerline_model[1] - vessel_centerline_model[0];
        else if (idx_on_cl == N - 1)
          vessel_tangent_model =
              vessel_centerline_model[N - 1] - vessel_centerline_model[N - 2];
        else
          vessel_tangent_model = vessel_centerline_model[idx_on_cl + 1] -
                                 vessel_centerline_model[idx_on_cl - 1];
      } else if (N >= 2) { // Fallback if point not exactly found or N small
        vessel_tangent_model =
            vessel_centerline_model.back() - vessel_centerline_model.front();
      } else {
        vessel_tangent_model = cv::Point2f(1, 0); // Default tangent
      }
      float norm_tangent = cv::norm(vessel_tangent_model);
      if (norm_tangent > 1e-6)
        vessel_tangent_model /= norm_tangent;
      else
        vessel_tangent_model = cv::Point2f(1, 0);

      auto tangent_line_opt = linePointSlopeToGeneral(
          vessel_tangent_point_model, vessel_tangent_model);
      if (!tangent_line_opt) {
        output.gate = getDefaultPWGate(input.roi);
        return output;
      }
      LineEquation vessel_tangent_line_model = *tangent_line_opt;
      LineEquation vessel_normal_line_model = getPerpendicular(
          vessel_tangent_line_model, vessel_tangent_point_model);

      LumenDiameterInfo ldi_model =
          getLumenDiameter(vessel_contours, vessel_normal_line_model,
                           vessel_centerline_model, {} // No plaques
          );
      if (ldi_model.isValid()) {
        calculated_lumen_infos_model.push_back(
            {ldi_model.diameter, ldi_model.diameter_endpoints,
             vessel_tangent_line_model, vessel_tangent_model});
      }
      // End of placeholder for skipped getNearestTangent
    }

    // Final PWGate construction from best calculated_lumen_info
    if (!calculated_lumen_infos_model.empty()) {
      // Python sorts and picks the first one (smallest diameter if plaque, or
      // the only one if no plaque) We already sorted if plaques were present.
      // If no plaques, it's just one.
      PWGateOutput::DebugLumenInfo best_lumen_info_model =
          calculated_lumen_infos_model[0];

      // Shift relevant points and lines to original ROI coordinates
      std::vector<Contour2f> diam_pts_model_list = {
          {best_lumen_info_model.diameter_points.first,
           best_lumen_info_model.diameter_points.second}};
      std::vector<Contour2f> shifted_diam_pts_list = shiftContourList(
          diam_pts_model_list, roi_move_xy, roi_scale_ratio_xy, move_first);

      if (shifted_diam_pts_list.empty() ||
          shifted_diam_pts_list[0].size() < 2) {
        output.gate = getDefaultPWGate(input.roi);
        return output;
      }

      std::pair<cv::Point2f, cv::Point2f> lumen_diameter_pts_roi = {
          shifted_diam_pts_list[0][0], shifted_diam_pts_list[0][1]};
      float lumen_diameter_roi = static_cast<float>(cv::norm(
          lumen_diameter_pts_roi.first - lumen_diameter_pts_roi.second));

      cv::Point2f pw_gate_point_roi =
          (lumen_diameter_pts_roi.first + lumen_diameter_pts_roi.second) * 0.5f;

      LineEquation vessel_tangent_line_roi =
          shiftLineEquation(best_lumen_info_model.tangent_line, roi_move_xy,
                            roi_scale_ratio_xy, move_first);
      float blood_angle_roi = getLineAngleDegrees(vessel_tangent_line_roi);

      output.gate = {pw_gate_point_roi, blood_angle_roi, lumen_diameter_roi,
                     ScanType::Longitudinal};

      // If debug, transform all infos
      if (input.debug) {
        for (const auto &info_model : calculated_lumen_infos_model) {
          std::vector<Contour2f> current_diam_pts_model = {
              {info_model.diameter_points.first,
               info_model.diameter_points.second}};
          std::vector<Contour2f> shifted_pts =
              shiftContourList(current_diam_pts_model, roi_move_xy,
                               roi_scale_ratio_xy, move_first);
          if (shifted_pts.empty() || shifted_pts[0].size() < 2)
            continue;

          std::pair<cv::Point2f, cv::Point2f> dbg_diam_pts_roi = {
              shifted_pts[0][0], shifted_pts[0][1]};
          double dbg_diam_roi =
              cv::norm(dbg_diam_pts_roi.first - dbg_diam_pts_roi.second);
          LineEquation dbg_tan_line_roi =
              shiftLineEquation(info_model.tangent_line, roi_move_xy,
                                roi_scale_ratio_xy, move_first);
          // Tangent vector itself doesn't shift, only scales (if not
          // normalized) and rotates. But its components scale if the coord
          // system scales. If (tx,ty) is a vector in model space. In ROI space,
          // if move_first=false: P1_roi = P1_model * scale + move P2_roi =
          // P2_model * scale + move Tangent_roi = P2_roi - P1_roi = (P2_model -
          // P1_model) * scale = Tangent_model * scale
          cv::Point2f dbg_tan_vec_roi =
              info_model
                  .tangent_vector; // This is unit vector, so should be
                                   // invariant to translation but scales if
                                   // axes scale differently. For simplicity,
                                   // let's re-derive from shifted line, or
                                   // assume it's used for angle only. If used
                                   // for angle, the shifted line is enough.
                                   // Python stores original tangent, it's
                                   // likely angle is from shifted line.

          output.debug_lumen_infos.push_back(
              {dbg_diam_roi, dbg_diam_pts_roi, dbg_tan_line_roi,
               info_model.tangent_vector /* or recomputed */});
        }
      }

    } else { // No valid lumen info calculated
      output.gate = getDefaultPWGate(input.roi);
    }
    return output;
  } // End Longitudinal case
}

std::pair<std::vector<double>, std::vector<std::pair<cv::Point2f, cv::Point2f>>>
PWSampler::computeShortestDistances(const Contour2f &contourA,
                                    const Contour2f &contourB) const {
  if (contourA.empty() || contourB.empty()) {
    return {{}, {}};
  }

  std::vector<double> min_distances;
  min_distances.reserve(contourB.size());
  std::vector<std::pair<cv::Point2f, cv::Point2f>> segments;
  segments.reserve(contourB.size());

  for (const auto &pt_b : contourB) {
    double current_min_dist_sq = std::numeric_limits<double>::max();
    cv::Point2f closest_pt_a;

    for (const auto &pt_a : contourA) {
      double dx = pt_a.x - pt_b.x;
      double dy = pt_a.y - pt_b.y;
      double dist_sq = dx * dx + dy * dy;
      if (dist_sq < current_min_dist_sq) {
        current_min_dist_sq = dist_sq;
        closest_pt_a = pt_a;
      }
    }
    min_distances.push_back(std::sqrt(current_min_dist_sq));
    segments.push_back({pt_b, closest_pt_a});
  }
  return {min_distances, segments};
}

Contour2f PWSampler::sortCurvePoints(const Contour2f &points) const {
  // Placeholder: return as is or throw error
  // This function is complex and involves KDTree, MST, DFS
  // For now, we'll just return the points unsorted or sorted by x then y as a
  // naive default if (points.size() < 3) return points; Contour2f sorted_pts =
  // points; std::sort(sorted_pts.begin(), sorted_pts.end(), [](const
  // cv::Point2f& a, const cv::Point2f& b){
  //     if (a.x != b.x) return a.x < b.x;
  //     return a.y < b.y;
  // });
  // return sorted_pts;
  // It's better to indicate it's not implemented if it's crucial for downstream
  // logic that relies on correct ordering.
  return points;
}

// find_line_curve_intersections - SKIPPED (uses CubicSpline)
std::vector<cv::Point2f>
PWSampler::findLineCurveIntersections(const Contour2f &curve,
                                      const LineEquation &line, int n_interp,
                                      double tol) const {
  std::cerr << "Warning: findLineCurveIntersections is not implemented in C++ "
               "(requires CubicSpline)."
            << std::endl;
  return {};
}

// get_contour_diameter_by_curve_normal_direction - SKIPPED (uses CubicSpline)
std::vector<ContourDiameterNormalInfo>
PWSampler::getContourDiameterByCurveNormalDirection(
    const Contour2f &polyline,
    const Contour2f &contour_b, // Python param is 'contour'
    double step, int window_size, bool only_longest) const {
  std::cerr << "Warning: getContourDiameterByCurveNormalDirection is not "
               "implemented in C++ (requires CubicSpline)."
            << std::endl;
  return {};
}

MaxDiameterInfo PWSampler::getContourMaxDiameter(
    const Contour2f &points) const { // Python: get_contour_max_diameter
  if (points.size() < 2) {
    return {{0, 0}, {0, 0}, 0.0};
  }

  double max_dist_sq = 0.0;
  cv::Point2f p1_max, p2_max;

  for (size_t i = 0; i < points.size(); ++i) {
    for (size_t j = i + 1; j < points.size(); ++j) {
      double dx = points[i].x - points[j].x;
      double dy = points[i].y - points[j].y;
      double dist_sq = dx * dx + dy * dy;
      if (dist_sq > max_dist_sq) {
        max_dist_sq = dist_sq;
        p1_max = points[i];
        p2_max = points[j];
      }
    }
  }
  return {p1_max, p2_max, std::sqrt(max_dist_sq)};
}

ClosestPointsInfo PWSampler::getClosestPoints(const Contour2f &contourA,
                                              const Contour2f &contourB) const {
  if (contourA.empty() || contourB.empty()) {
    return {std::numeric_limits<double>::max(), {0, 0}, {0, 0}};
  }

  double min_dist_sq = std::numeric_limits<double>::max();
  cv::Point2f final_pointA, final_pointB;

  // Python version has an interpolation part for point B.
  // Skipping interpolation for now, just finding closest pair of vertices.
  for (const auto &pA : contourA) {
    for (const auto &pB : contourB) {
      double dx = pA.x - pB.x;
      double dy = pA.y - pB.y;
      double dist_sq = dx * dx + dy * dy;
      if (dist_sq < min_dist_sq) {
        min_dist_sq = dist_sq;
        final_pointA = pA;
        final_pointB = pB;
      }
    }
  }
  return {std::sqrt(min_dist_sq), final_pointA, final_pointB};
}

Contour2f PWSampler::getContourPointsWithinDistance(const Contour2f &contourA,
                                                    const cv::Point2f &P,
                                                    double max_distance) const {
  Contour2f result_points;
  if (contourA.empty()) {
    return result_points;
  }
  double max_dist_sq = max_distance * max_distance;

  for (const auto &ptA : contourA) {
    double dx = ptA.x - P.x;
    double dy = ptA.y - P.y;
    if (dx * dx + dy * dy <= max_dist_sq) {
      result_points.push_back(ptA);
    }
  }
  return result_points;
}

std::optional<cv::Point2f> PWSampler::findIntersectionPoint(
    const Contour2f &contour,
    const std::pair<cv::Point2f, cv::Point2f> &ray) const {
  if (contour.size() < 2)
    return std::nullopt;

  cv::Point2f O = ray.first;
  cv::Point2f D = ray.second - O; // Direction vector of the ray

  double min_t = std::numeric_limits<double>::max();
  cv::Point2f intersection_point;
  bool found = false;

  for (size_t i = 0; i < contour.size(); ++i) {
    cv::Point2f p1 = contour[i];
    cv::Point2f p2 =
        contour[(i + 1) %
                contour.size()]; // Next point, wraps around for closed contour
    cv::Point2f E = p2 - p1;     // Edge vector

    // D.cross(E) = D.x * E.y - D.y * E.x;
    double det = D.x * E.y - D.y * E.x;

    constexpr double epsilon = 1e-10;
    if (std::abs(det) < epsilon) { // Lines are parallel or collinear
      continue;
    }

    // (p1 - O).cross(E)
    double t_num = (p1.x - O.x) * E.y - (p1.y - O.y) * E.x;
    // (p1 - O).cross(D)
    double s_num = (p1.x - O.x) * D.y - (p1.y - O.y) * D.x;

    double t = t_num / det;
    double s = s_num / det;

    if (t >= 0 && s >= 0 && s <= 1) { // Intersection on ray and on segment
      if (t < min_t) {
        min_t = t;
        intersection_point = O + D * static_cast<float>(t);
        found = true;
      }
    }
  }

  if (found) {
    // Python version rounds to 10 decimal places.
    // For cv::Point2f, direct assignment is usually fine.
    return intersection_point;
  }
  return std::nullopt;
}

std::pair<Contour2f, Contour2f> PWSampler::splitPointsByCurve(
    const Contour2f &curve,           // Python 'curv'
    const Contour2f &points_to_split, // Python 'points'
    int k_neighbors) const {          // Python 'k'
  if (curve.size() < 2 || points_to_split.empty()) {
    return {{}, {}};
  }

  Contour2f left_side, right_side;
  left_side.reserve(points_to_split.size());
  right_side.reserve(points_to_split.size());

  // Precompute normals for curve points (can be optimized if curve is static
  // and reused)
  std::vector<cv::Point2f> curve_normals(curve.size());
  int half_k = k_neighbors / 2;

  for (int i = 0; i < static_cast<int>(curve.size()); ++i) {
    int start_idx = std::max(0, i - half_k);
    // Python's end_indices = np.minimum(len(A)-1, indices + k//2 + 1)
    // For a segment from start to end, it's (k+1) points if k is even.
    // Let's use k points around `i`, so `i - k/2` to `i + k/2`.
    // Python's `end_points = A[end_indices]` implies indices + k//2 + 1 is
    // exclusive, or k//2 is radius. `end_indices = np.minimum(len(A)-1, indices
    // + k//2)` if k//2 is radius and indices + k//2 is inclusive. Let's assume
    // k_neighbors is the total number of points to consider for tangent. Python
    // original: start_indices = np.maximum(0, indices - k//2)
    //                   end_indices = np.minimum(len(A)-1, indices + k//2 + 1)
    // This is confusing. Let's try simpler: use `i-half_k` and `i+half_k`.
    int end_idx = std::min(static_cast<int>(curve.size()) - 1, i + half_k);

    // Ensure start_idx is before end_idx for a valid tangent calculation
    if (start_idx >= end_idx &&
        curve.size() >
            1) { // If not enough points, try using first/last segment
      if (i == 0) {
        start_idx = 0;
        end_idx = std::min(1, (int)curve.size() - 1);
      } else {
        end_idx = (int)curve.size() - 1;
        start_idx = std::max(0, end_idx - 1);
      }
    }
    if (start_idx >= end_idx) {             // single point curve or logic error
      curve_normals[i] = cv::Point2f(0, 1); // Default normal
      continue;
    }

    cv::Point2f tangent_vec = curve[end_idx] - curve[start_idx];
    float norm = cv::norm(tangent_vec);
    if (norm > 1e-6) {
      tangent_vec /= norm;
      curve_normals[i] =
          cv::Point2f(-tangent_vec.y, tangent_vec.x); // Left normal
    } else {
      // Default normal or handle as error/special case
      // For a horizontal segment, tangent (dx, 0), normal (0, dx) or (0,1) if
      // dx > 0 For a vertical segment, tangent (0, dy), normal (-dy, 0) or
      // (-1,0) if dy > 0 If it's a duplicate point, this is problematic. Use
      // previous normal if available, or a default.
      if (i > 0)
        curve_normals[i] = curve_normals[i - 1];
      else
        curve_normals[i] = cv::Point2f(0, 1); // Default up
    }
  }

  for (const auto &p_split : points_to_split) {
    // 1. Find nearest point on curve and its index
    double min_dist_sq = std::numeric_limits<double>::max();
    int nearest_idx = 0;
    for (int i = 0; i < static_cast<int>(curve.size()); ++i) {
      double dx = p_split.x - curve[i].x;
      double dy = p_split.y - curve[i].y;
      double dist_sq = dx * dx + dy * dy;
      if (dist_sq < min_dist_sq) {
        min_dist_sq = dist_sq;
        nearest_idx = i;
      }
    }

    cv::Point2f nearest_curve_pt = curve[nearest_idx];
    cv::Point2f normal_at_nearest = curve_normals[nearest_idx];

    cv::Point2f vec_to_curve = p_split - nearest_curve_pt;
    double dot_product = vec_to_curve.dot(normal_at_nearest);

    if (dot_product > 0) { // Python: dots > 0 for left_mask
      left_side.push_back(p_split);
    } else {
      right_side.push_back(p_split);
    }
  }
  return {left_side, right_side};
}

std::pair<Contour2f, Contour2f>
PWSampler::splitPointsByLine(const Contour2f &points_on_line,
                             const cv::Point2f &pivot_point,
                             const LineEquation &line) const {
  Contour2f positive_side, negative_side;
  if (points_on_line.empty()) {
    return {positive_side, negative_side};
  }
  // Direction vector along the line (or perpendicular to the normal (a,b))
  // Python uses (b, -a). This vector is parallel to ax+by+c=0
  cv::Point2f direction_vec(static_cast<float>(line.b),
                            static_cast<float>(-line.a));

  for (const auto &pt : points_on_line) {
    cv::Point2f delta = pt - pivot_point;
    double projection = delta.dot(direction_vec);
    if (projection > 0) {
      positive_side.push_back(pt);
    } else {
      negative_side.push_back(
          pt); // Includes points exactly on pivot (projection == 0)
    }
  }
  return {positive_side, negative_side};
}

std::optional<LineEquation> PWSampler::getGeneralLine(
    const cv::Point2f &p1, const std::optional<cv::Point2f> &p2_opt,
    const std::optional<cv::Point2f> &tangent_vector_opt) const {
  if (p2_opt) {
    return lineTwoPointsToGeneral(p1, *p2_opt);
  }
  if (tangent_vector_opt) {
    return linePointSlopeToGeneral(p1, *tangent_vector_opt);
  }
  return std::nullopt; // Should not happen based on Python's assert
}

std::optional<LineEquation>
PWSampler::linePointSlopeToGeneral(const cv::Point2f &point,
                                   const cv::Point2f &tangent_vector) const {
  // tangent (tx, ty)
  // normal is (-ty, tx) or (ty, -tx). Let's use (ty, -tx) for ax+by+c=0
  // a = ty, b = -tx
  // c = -(a*x1 + b*y1) = -(ty*x1 - tx*y1) = tx*y1 - ty*x1
  if (tangent_vector.x == 0 && tangent_vector.y == 0)
    return std::nullopt; // degenerate tangent
  return LineEquation{tangent_vector.y, -tangent_vector.x,
                      tangent_vector.x * point.y - tangent_vector.y * point.x};
}

LineEquation
PWSampler::lineTwoPointsToGeneral(const cv::Point2f &point1,
                                  const cv::Point2f &point2) const {
  double x1 = point1.x, y1 = point1.y;
  double x2 = point2.x, y2 = point2.y;

  // Line equation: (y1-y2)x + (x2-x1)y + (x1y2 - x2y1) = 0
  // a = y1-y2
  // b = x2-x1
  // c = x1y2 - x2y1
  // Python version has: (y2-y1, x1-x2, x2*y1 - x1*y2)
  // This is equivalent by multiplying by -1. Let's stick to Python's version
  // for consistency.
  double a = y2 - y1;
  double b = x1 - x2;
  double c = x2 * y1 - x1 * y2;

  if (std::abs(x1 - x2) < 1e-9) { // Vertical line
    return {1.0, 0.0, -x1};
  }
  if (std::abs(y1 - y2) < 1e-9) { // Horizontal line
    return {0.0, 1.0, -y1};
  }
  return {a, b, c};
}

// get_nearest_tangent - SKIPPED (uses CubicSpline)
std::optional<NearestTangentInfo>
PWSampler::getNearestTangent(const Contour2f &polyline,
                             const cv::Point2f &target_point, int window_size,
                             int n_interp, double eps) const {
  std::cerr << "Warning: getNearestTangent is not implemented in C++ (requires "
               "CubicSpline)."
            << std::endl;
  return std::nullopt;
}

LineEquation PWSampler::getPerpendicular(const LineEquation &line,
                                         const cv::Point2f &point) const {
  // Original line: ax + by + c = 0. Slope m = -a/b. Normal vector (a,b).
  // Direction vector (-b,a) or (b,-a). Perpendicular line has slope m_perp =
  // -1/m = b/a. Equation: y - y0 = (b/a)(x - x0) a(y - y0) = b(x - x0) ay - ay0
  // = bx - bx0 bx - ay + (ay0 - bx0) = 0 So, a_perp = b, b_perp = -a, c_perp =
  // a*y0 - b*x0.
  return {line.b, -line.a, line.a * point.y - line.b * point.x};
}

bool PWSampler::isContourUpperSideOfCurve(
    const Contour2f &contour_to_check, const Contour2f &reference_curve) const {
  if (contour_to_check.empty())
    return false; // Or based on convention
  auto [upper_points, lower_points] =
      splitPointsByCurve(reference_curve, contour_to_check);
  return upper_points.size() > lower_points.size();
}

LumenDiameterInfo PWSampler::getLumenDiameter(
    const std::vector<Contour2f> &vessel_contours,
    const LineEquation &normal_line, const Contour2f &centerline,
    const std::vector<Contour2f> &pleque_contours) const {
  Contour2f vessel_boundary_pts;
  for (const auto &vessel_contour : vessel_contours) {
    // findLineCurveIntersections is SKIPPED. This function depends on it.
    std::vector<cv::Point2f> intersections =
        findLineCurveIntersections(vessel_contour, normal_line);
    vessel_boundary_pts.insert(vessel_boundary_pts.end(), intersections.begin(),
                               intersections.end());
  }

  if (vessel_boundary_pts.empty() &&
      pleque_contours
          .empty()) { // If no intersections from vessel and no plaques
    std::cerr << "Warning: getLumenDiameter cannot proceed without "
                 "findLineCurveIntersections results."
              << std::endl;
    return {-1.0, {{0, 0}, {0, 0}}};
  }

  auto [upper_boundary_pts_list, lower_boundary_pts_list] =
      splitPointsByCurve(centerline, vessel_boundary_pts);

  if (!pleque_contours.empty()) {
    for (const auto &pleque_contour : pleque_contours) {
      // This also depends on findLineCurveIntersections
      std::vector<cv::Point2f> pleque_intersections =
          findLineCurveIntersections(pleque_contour, normal_line);
      if (isContourUpperSideOfCurve(pleque_contour, centerline)) {
        upper_boundary_pts_list.insert(upper_boundary_pts_list.end(),
                                       pleque_intersections.begin(),
                                       pleque_intersections.end());
      } else {
        lower_boundary_pts_list.insert(lower_boundary_pts_list.end(),
                                       pleque_intersections.begin(),
                                       pleque_intersections.end());
      }
    }
  }

  if (upper_boundary_pts_list.empty() || lower_boundary_pts_list.empty()) {
    if (findLineCurveIntersections({}, {}).empty()) { // Check if stub was hit
      std::cerr << "Warning: getLumenDiameter results may be invalid due to "
                   "skipped findLineCurveIntersections."
                << std::endl;
    }
    return {-1.0, {{0, 0}, {0, 0}}}; // Return invalid
  }

  ClosestPointsInfo cpi =
      getClosestPoints(upper_boundary_pts_list, lower_boundary_pts_list);
  return {cpi.distance, {cpi.point_on_A, cpi.point_on_B}};
}

PWGate PWSampler::getDefaultPWGate(const cv::Rect &roi) const {
  cv::Point2f center_pt(roi.x + roi.width / 2.0f, roi.y + roi.height / 2.0f);
  return {center_pt, 0.0f, -1.0f, ScanType::Others};
}

cv::Mat PWSampler::getMaxConnectedComponentSeg(const cv::Mat &mask) const {
  if (mask.empty() || mask.type() != CV_8U) {
    std::cerr
        << "getMaxConnectedComponentSeg: Input mask is empty or not CV_8U."
        << std::endl;
    return cv::Mat();
  }
  cv::Mat labels, stats, centroids;
  // Ensure mask is binary for connectedComponents
  cv::Mat binary_mask = mask > 0;
  int num_labels = cv::connectedComponentsWithStats(binary_mask, labels, stats,
                                                    centroids, 8, CV_32S);

  if (num_labels <= 1) {
    return cv::Mat();
  }

  int max_area = 0;
  int max_label = 0;
  for (int i = 1; i < num_labels; ++i) {
    int area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (area > max_area) {
      max_area = area;
      max_label = i;
    }
  }

  cv::Mat result_mask = cv::Mat::zeros(mask.size(), CV_8U);
  if (max_label > 0) {
    /**
    Create a binary mask of the largest component If original mask was a label
    mask and we need to preserve original label values: mask.copyTo(result_mask,
    labels == max_label); Python: mask[relabeled_mask != max_area_label] = 0.
    This implies modifying the input or a copy. Let's assume the input mask to
    this function is binary (e.g., seg_mask > 0) If seg_mask itself is a
    multi-label mask, and we want to keep only largest object *from that
    multi-label mask* then the logic would be: cv::Mat output_mask =
    mask.clone(); output_mask.setTo(0, labels != max_label); return output_mask;
    Given python `mask[relabeled_mask != max_area_label] = 0`, it seems it
    modifies a copy of the original mask. And the input `mask` to python
    `get_max_cc_seg` is `seg_mask`. So, we should return a mask that is 0
    everywhere except for the largest component, where it retains original
    values from `mask`.
     */
    result_mask = (labels == max_label) * 255;
    cv::imwrite("vis_max_cc.png", result_mask);
    cv::Mat output_mask = cv::Mat::zeros(mask.size(), mask.type());
    mask.copyTo(output_mask, labels == max_label);
    return output_mask;
  }
  return cv::Mat(); // Should not happen if num_labels > 1
}

LineEquation PWSampler::shiftLineEquation(
    const LineEquation &line,
    const cv::Point2f &move_xy,          // (m, n) in Python
    const cv::Point2f &resize_ratios_xy, // (p, q) in Python
    bool move_first) const {
  double a = line.a, b = line.b, c = line.c;
  double m = move_xy.x, n = move_xy.y;                   // dx, dy
  double p = resize_ratios_xy.x, q = resize_ratios_xy.y; // sx, sy

  // Original line: ax + by + c = 0
  if (move_first) {
    // 1. Translate: x' = x - m, y' = y - n => x = x' + m, y = y' + n
    // a(x'+m) + b(y'+n) + c = 0
    // ax' + by' + (am + bn + c) = 0.  Let c_prime = am + bn + c.
    // So, a_temp = a, b_temp = b, c_temp = am + bn + c
    double c_prime =
        c + a * m +
        b * n; // Python: c - a*m - b*n. This depends on transform definition.
               // If x_new = x_old + m => x_old = x_new - m.
               // a(x_new - m) + b(y_new - n) + c = 0
               // ax_new + by_new + (c - am - bn) = 0. This matches Python.
    c_prime = c - a * m - b * n;

    // 2. Scale: x'' = x'/p, y'' = y'/q => x' = x''*p, y' = y''*q
    // a(x''p) + b(y''q) + c_prime = 0
    // (ap)x'' + (bq)y'' + c_prime = 0
    // Python: a_transformed = a * q, b_transformed = b * p, c_transformed =
    // c_prime * p * q This implies a coordinate system definition or equation
    // manipulation difference. Let's re-derive based on standard transform:
    // x_new = p * (x_old + m), y_new = q * (y_old + n) Or: x_new = p * x_old +
    // m_eff, y_new = q * y_old + n_eff (if scaling center is origin)

    // Python logic:
    // c_prime = c - a * m - b * n;
    // a_transformed = a * q; b_transformed = b * p; c_transformed = c_prime * p
    // * q; This implies transformation x_new = x_old/p, y_new = y_old/q for the
    // scaling step. And the coefficients of line transform as a'/p, b'/q,
    // c'/(pq) for ax+by+c=0 Or if line is aX+bY+C=0 and X=px, Y=qy, then
    // a(px)+b(qy)+C=0 => (ap)x + (bq)y + C=0. If the *coordinates* are scaled
    // by (p,q) i.e. x_new = x_old*p, y_new = y_old*q Then substitute x_old =
    // x_new/p, y_old = y_new/q into a*x_old + b*y_old + c_prime = 0 (a/p)x_new
    // + (b/q)y_new + c_prime = 0. Multiply by pq: (aq)x_new + (bp)y_new +
    // c_prime*p*q = 0. This matches Python's `a_transformed = a*q`,
    // `b_transformed = b*p`, `c_transformed = c_prime*p*q`.
    double a_transformed = a * q;
    double b_transformed = b * p;
    double c_transformed = c_prime * p * q;
    return {a_transformed, b_transformed, c_transformed};

  } else { // Scale first, then move
    // 1. Scale: x' = x/p, y' = y/q => x = x'p, y = y'q (using Python's implied
    // scaling of coordinates) a(x'p) + b(y'q) + c = 0 (ap)x' + (bq)y' + c = 0.
    // Python: a_scaled = a / p, b_scaled = b / q. This implies coefficients
    // transform like this. This means points transform: x_new = x_old / p,
    // y_new = y_old / q Then x_old = x_new * p, y_old = y_new * q. a * (x_new *
    // p) + b * (y_new * q) + c = 0. (a*p) x_new + (b*q) y_new + c = 0. Let's
    // use Python's direct coefficient transformation for now, assuming it's
    // tested. a_scaled = a / p, b_scaled = b / q, c_scaled = c (implicitly, not
    // changed yet)
    double a_scaled = a / p; // careful with p=0
    double b_scaled = b / q; // careful with q=0
    if (std::abs(p) < 1e-9 || std::abs(q) < 1e-9) {
      std::cerr
          << "Warning: Division by zero in shiftLineEquation (scale first)."
          << std::endl;
      // return original line or throw
      return line;
    }
    double c_temp =
        c; // c is not scaled in this step by Python logic, only a and b

    // 2. Translate: x'' = x' - m, y'' = y' - n => x' = x''+m, y' = y''+n
    // a_scaled(x''+m) + b_scaled(y''+n) + c_temp = 0
    // a_scaled*x'' + b_scaled*y'' + (a_scaled*m + b_scaled*n + c_temp) = 0
    // Python: c_translated = c - a_scaled * m - b_scaled * n
    // Again, this implies x_new = x_old - m, y_new = y_old - n for points.
    // So coefficients a,b are unchanged, c becomes c' = c - a*m - b*n.
    double c_translated = c_temp - a_scaled * m - b_scaled * n;
    return {a_scaled, b_scaled, c_translated};
  }
}

std::vector<std::vector<double>>
PWSampler::calculatePointsDistanceMatrix(const Contour2f &points) const {
  size_t n = points.size();
  std::vector<std::vector<double>> dist_matrix(n, std::vector<double>(n, 0.0));
  if (n == 0)
    return dist_matrix;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i; j < n; ++j) { // Symmetric matrix
      if (i == j) {
        dist_matrix[i][j] = 0.0;
      } else {
        double d = cv::norm(points[i] - points[j]);
        dist_matrix[i][j] = d;
        dist_matrix[j][i] = d;
      }
    }
  }
  return dist_matrix;
}

// ------ The following are analogous to
// us_pipeline.carotis.modules.measure_utils ------ or other utility modules
// from the Python imports. These are placeholders or simplified.

// Corresponds to us_pipeline.carotis.modules.measure_utils.get_coord_shift
// Python returns (yx_move, yx_scale_ratio, move_first)
// Let's make C++ use xy: (xy_move, xy_scale_ratio, move_first)
std::tuple<cv::Point2f, cv::Point2f, bool>
PWSampler::getCoordShift(const cv::Size &model_size,
                         const cv::Rect &roi) const {
  // This function's logic is specific to how `us_pipeline` handles ROI vs model
  // coordinates. A common scenario: model_size is the network input size,
  // roi is the region in the original image. If seg_mask is output of
  // model on a *cropped and resized* version of roi: Example: original
  // image -> roi crop -> resize to model_size -> network -> seg_mask (size
  // model_size) To map points from seg_mask (model coords) back to original
  // image coords (within ROI):
  // 1. Un-resize: pt_roi.x = pt_model.x * (roi.width / model_size.width)
  //              pt_roi.y = pt_model.y * (roi.height / model_size.height)
  // 2. Un-crop (add ROI offset): pt_orig.x = pt_roi.x + roi.x
  //                               pt_orig.y = pt_roi.y + roi.y
  // This means: move = (roi.x, roi.y), resize =
  // (roi.width/model_size.width, roi.height/model_size.height) And
  // move_first should be FALSE (scale then translate). pt_orig = (pt_model *
  // resize_ratio) + move_offset

  // The Python code `line_shifing` and `contour_list_shifting` seem to use:
  // new_coord = (old_coord + move) * resize OR new_coord = old_coord * resize +
  // move If mapping FROM model TO original image: old_coord = model_coord,
  // new_coord = original_image_coord original_image_coord.x = (model_coord.x *
  // scale_x) + offset_x original_image_coord.y = (model_coord.y * scale_y) +
  // offset_y Here, scale_x = roi.width / (float)model_size.width
  //       scale_y = roi.height / (float)model_size.height
  //       offset_x = roi.x
  //       offset_y = roi.y
  // This transformation corresponds to move_first = false (scale first, then
  // add offset). The `move` in `contour_list_shifting` would be `(offset_x,
  // offset_y)` The `resize` in `contour_list_shifting` would be `(scale_x,
  // scale_y)`

  // Python's `get_coord_shift` returns `roi_move`, `roi_scale_ratio`,
  // `move_first`. If `roi_move` is (dy, dx) and `roi_scale_ratio` is (sy, sx)
  // from Python. Let's assume this a common implementation.
  cv::Point2f roi_move(static_cast<float>(roi.x), static_cast<float>(roi.y));
  cv::Point2f roi_scale_ratio(1.0f, 1.0f);
  if (model_size.width > 0 && model_size.height > 0) {
    roi_scale_ratio.x = static_cast<float>(roi.width) / model_size.width;
    roi_scale_ratio.y = static_cast<float>(roi.height) / model_size.height;
  }
  bool move_first = false; // Typically scale (from model to ROI sub-image) then
                           // translate (to full image coords)

  // The Python code's `contour_list_shifting` with `move_first=True` uses:
  // shifted_pt = (pt + move) * resize_ratio
  // With `move_first=False` uses:
  // shifted_pt = pt * resize_ratio + move

  // If the Python `get_coord_shift` means:
  // `roi_move`: translation applied in the *source* coordinate system (e.g.
  // model output space) `roi_scale_ratio`: scaling applied to the *source*
  // coordinate system And the goal is to map source (model) points to target
  // (original image ROI) points. If model output is at (0,0) origin for the
  // processed ROI:
  //   move = (roi.x, roi.y)
  //   scale = (roi.width/model_width, roi.height/model_height)
  //   move_first = false (scale model points, then add roi offset)
  // This seems the most standard interpretation for vision pipelines.
  // Python code uses `roi_move[::-1]` and `roi_scale_ratio[::-1]` when calling
  // `contour_list_shifting`, implying `get_coord_shift` might return in YX
  // order. We are using XY order here.

  std::cout << "Info: getCoordShift using default logic (scale then translate "
               "for model->original)."
            << std::endl;
  return {roi_move, roi_scale_ratio, move_first};
}

// Corresponds to us_pipeline.carotis.modules.measure_utils.find_contours
// Simplified version. Python version has `only_largest` logic.
std::vector<Contour> PWSampler::findContoursFromMask(const cv::Mat &binary_mask,
                                                     int contour_mode,
                                                     int contour_method,
                                                     bool largest_only) const {
  std::vector<Contour> contours;
  if (binary_mask.empty() || binary_mask.type() != CV_8U) {
    std::cerr << "findContoursFromMask: Input mask is empty or not CV_8U."
              << std::endl;
    return contours;
  }
  cv::findContours(binary_mask, contours, contour_mode, contour_method);

  if (largest_only && !contours.empty()) {
    auto largest_it =
        std::max_element(contours.begin(), contours.end(),
                         [](const Contour &a, const Contour &b) {
                           return cv::contourArea(a) < cv::contourArea(b);
                         });
    return {*largest_it};
  }
  return contours;
}

std::pair<std::vector<Contour>, cv::Mat>
PWSampler::findContours(const cv::Mat &mask, int contour_mode,
                        int contour_method, bool only_largest) const {
  // Ensure the input mask is of the correct type (CV_8U) for cv::findContours
  if (mask.empty() || mask.channels() != 1) {
    // Handle error: mask must be a single-channel image.
    // You might want to throw an exception or print an error.
    std::cerr
        << "Error in findContours: Input mask is empty or not single-channel."
        << std::endl;
    return {{}, cv::Mat()};
  }

  cv::Mat processed_mask;
  if (mask.type() != CV_8U) {
    // Python's mask.astype(np.uint8) equivalent
    mask.convertTo(processed_mask, CV_8U);
  } else {
    processed_mask = mask;
  }

  std::vector<Contour> contours;
  cv::Mat hierarchy;
  cv::findContours(processed_mask, contours, hierarchy, contour_mode,
                   contour_method);

  if (only_largest && contours.size() > 1) {
    // Find the iterator to the element with the largest area
    auto largest_it =
        std::max_element(contours.begin(), contours.end(),
                         [](const Contour &a, const Contour &b) {
                           return cv::contourArea(a) < cv::contourArea(b);
                         });

    // Return a new vector containing only the largest contour
    return {{*largest_it}, hierarchy};
  }

  // Return all contours if not `only_largest` or if there's 0 or 1 contour
  return {contours, hierarchy};
}

std::pair<std::vector<Contour>, std::vector<double>>
PWSampler::getContoursAndFilterByArea(const cv::Mat &mask, int contour_mode,
                                      int contour_method, bool only_largest,
                                      float area_ratio_th, int area_th) const {
  // Step 1: Find initial contours using our helper function
  auto [contours, hierarchy] =
      findContours(mask, contour_mode, contour_method, only_largest);

  if (contours.empty()) {
    return {{}, {}};
  }

  // Step 2: Calculate areas for all found contours
  std::vector<double> areas;
  areas.reserve(contours.size());
  for (const auto &c : contours) {
    areas.push_back(cv::contourArea(c));
  }

  // Step 3: Determine thresholds and prepare for filtering
  double min_relative_area = -1.0;
  bool apply_ratio_filter = (area_ratio_th > 0.0f && area_ratio_th <= 1.0f);

  if (apply_ratio_filter) {
    // Calculate sum of all contour areas
    double total_contour_area =
        std::accumulate(areas.begin(), areas.end(), 0.0);
    if (total_contour_area > 0) {
      min_relative_area = total_contour_area * area_ratio_th;
    }
  }

  double min_absolute_area = static_cast<double>(area_th);
  bool apply_abs_filter = (area_th >= 0);

  // Step 4: Filter the contours and areas in a single pass
  std::vector<Contour> filtered_contours;
  std::vector<double> filtered_areas;

  for (size_t i = 0; i < contours.size(); ++i) {
    bool passes_ratio = !apply_ratio_filter || (areas[i] >= min_relative_area);
    bool passes_abs = !apply_abs_filter || (areas[i] >= min_absolute_area);

    if (passes_ratio && passes_abs) {
      filtered_contours.push_back(contours[i]);
      filtered_areas.push_back(areas[i]);
    }
  }

  return {filtered_contours, filtered_areas};
}

// Corresponds to us_pipeline.carotis.modules.measure_utils.get_frame_scan_type
ScanType PWSampler::getFrameScanType(
    const std::vector<Contour> &vessel_contours,
    const std::vector<double>
        &vessel_contour_areas, // Python doesn't seem to use areas here
    float longitudinal_section_aspect_ratio_th,
    float longitudinal_section_eccentricity_th) const {
  if (vessel_contours.empty()) {
    return ScanType::Others;
  }
  // Python's get_frame_scan_type takes vessel_contour_list and
  // vessel_contour_area_list It then iterates through them. Let's assume we
  // check the largest one if multiple. Or, if any satisfies longitudinal, it's
  // longitudinal. The Python `get_pw_gate` calls
  // `get_contour_and_filter_by_area` with `only_largest=True` for vessel
  // contours, so `vessel_contours` will have at most one element.

  for (const auto &contour : vessel_contours) {
    if (contour.size() < 5) { // Need at least 5 points for fitEllipse
      // Treat as Others if too few points for shape analysis
      continue;
    }
    cv::RotatedRect rrect = cv::fitEllipse(contour);
    float w = rrect.size.width;
    float h = rrect.size.height;

    float aspect_ratio =
        (w > h) ? (w / std::max(h, 1e-6f)) : (h / std::max(w, 1e-6f));

    // Eccentricity e = sqrt(1 - (b/a)^2) where a is major, b is minor axis
    float major_axis = std::max(w, h);
    float minor_axis = std::min(w, h);
    float eccentricity = 0.0f;
    if (major_axis > 1e-6) {
      eccentricity = std::sqrt(1.0f - (minor_axis * minor_axis) /
                                          (major_axis * major_axis));
    }

    // Python logic:
    // if aspect_ratio > th1 and eccentricity > th2: return Longitudinal
    // Here, longitudinal_section_eccentricity_th might be a minimum.
    // High aspect ratio (elongated) AND high eccentricity (elliptical, not
    // circular) -> Longitudinal
    if (aspect_ratio > longitudinal_section_aspect_ratio_th &&
        eccentricity > longitudinal_section_eccentricity_th) {
      return ScanType::Longitudinal;
    }
  }
  return ScanType::Others;
}

// Corresponds to
// us_pipeline.carotis.modules.measure_utils.contour_list_shifting Overload for
// Contour (vector<cv::Point>)
std::vector<Contour2f> PWSampler::shiftContourList(
    const std::vector<Contour> &contours, const cv::Point2f &move_xy,
    const cv::Point2f &resize_ratios_xy, bool move_first) const {
  std::vector<Contour2f> shifted_contour_list;
  shifted_contour_list.reserve(contours.size());
  for (const auto &contour : contours) {
    Contour2f shifted_contour;
    shifted_contour.reserve(contour.size());
    for (const auto &pt_i : contour) {
      cv::Point2f pt(static_cast<float>(pt_i.x), static_cast<float>(pt_i.y));
      cv::Point2f shifted_pt;
      if (move_first) {
        shifted_pt.x = (pt.x + move_xy.x) * resize_ratios_xy.x;
        shifted_pt.y = (pt.y + move_xy.y) * resize_ratios_xy.y;
      } else {
        shifted_pt.x = pt.x * resize_ratios_xy.x + move_xy.x;
        shifted_pt.y = pt.y * resize_ratios_xy.y + move_xy.y;
      }
      shifted_contour.push_back(shifted_pt);
    }
    shifted_contour_list.push_back(shifted_contour);
  }
  return shifted_contour_list;
}
// Overload for Contour2f (vector<cv::Point2f>)
std::vector<Contour2f> PWSampler::shiftContourList(
    const std::vector<Contour2f> &contours, const cv::Point2f &move_xy,
    const cv::Point2f &resize_ratios_xy, bool move_first) const {
  std::vector<Contour2f> shifted_contour_list;
  shifted_contour_list.reserve(contours.size());
  for (const auto &contour : contours) {
    Contour2f shifted_contour;
    shifted_contour.reserve(contour.size());
    for (const auto &pt : contour) {
      cv::Point2f shifted_pt;
      if (move_first) {
        shifted_pt.x = (pt.x + move_xy.x) * resize_ratios_xy.x;
        shifted_pt.y = (pt.y + move_xy.y) * resize_ratios_xy.y;
      } else {
        shifted_pt.x = pt.x * resize_ratios_xy.x + move_xy.x;
        shifted_pt.y = pt.y * resize_ratios_xy.y + move_xy.y;
      }
      shifted_contour.push_back(shifted_pt);
    }
    shifted_contour_list.push_back(shifted_contour);
  }
  return shifted_contour_list;
}

// Corresponds to us_pipeline.carotis.modules.centerline_utils.get_centerline
Contour2f PWSampler::getCenterline(const cv::Mat &seg_mask,
                                   int centerline_label,
                                   bool order_points) const {
  // This is a complex function, likely involving skeletonization or other
  // medial axis transforms. Placeholder implementation.
  std::cerr
      << "Warning: getCenterline is a placeholder and not fully implemented."
      << std::endl;
  // Example: Return the contour of the whole mask as a pseudo-centerline for
  // testing std::vector<Contour> contours = findContoursFromMask(seg_mask > 0,
  // cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, true); if (!contours.empty())
  // return convertContourTo2f(contours[0]);
  return {};
}

// Corresponds to us_pipeline.carotis.modules.blood_gate_utils.get_line_degree
// Angle definition from PWGate:
// x正方向旋转至目标方向的角度，顺时针方向为正，逆时针方向为负，取值为[-180,
// 180]
float PWSampler::getLineAngleDegrees(const LineEquation &line) const {
  // Line: ax + by + c = 0
  // Normal vector N = (a,b)
  // Direction vector D = (-b, a) or (b, -a). Let's use D = (-b, a) for standard
  // orientation. Angle of D with x-axis: atan2(a, -b) Python's
  // `get_line_degree` might have its own conventions. Let tangent vector be
  // (tx, ty). Angle from x-axis is atan2(ty, tx). If line is ax+by+c=0, a
  // tangent is (b, -a) or (-b, a). Let's use tx = b, ty = -a. Angle_rad =
  // atan2(-a, b) (counter-clockwise from +x axis to vector (b, -a))
  double angle_rad =
      std::atan2(-line.a, line.b); // Angle for direction vector (b, -a)
  double angle_deg = angle_rad * 180.0 / M_PI;

  // PWGate definition: "顺时针方向为正" (clockwise positive)
  // Standard atan2 gives counter-clockwise positive. So we need to negate.
  angle_deg = -angle_deg;

  // Normalize to [-180, 180]
  while (angle_deg > 180.0)
    angle_deg -= 360.0;
  while (angle_deg <= -180.0)
    angle_deg +=
        360.0; // Use <= for -180 case to become +180 if that's the convention

  return static_cast<float>(angle_deg);
}

Contour2f PWSampler::convertContourTo2f(const Contour &int_contour) const {
  Contour2f float_contour;
  float_contour.reserve(int_contour.size());
  for (const auto &pt : int_contour) {
    float_contour.emplace_back(static_cast<float>(pt.x),
                               static_cast<float>(pt.y));
  }
  return float_contour;
}

Contour PWSampler::convertContourTo2i(const Contour2f &float_contour) const {
  Contour int_contour;
  int_contour.reserve(float_contour.size());
  for (const auto &pt : float_contour) {
    int_contour.emplace_back(cv::saturate_cast<int>(pt.x),
                             cv::saturate_cast<int>(pt.y));
  }
  return int_contour;
}

std::vector<Contour2f> PWSampler::convertContourListTo2f(
    const std::vector<Contour> &int_contours) const {
  std::vector<Contour2f> float_contours;
  float_contours.reserve(int_contours.size());
  for (const auto &c : int_contours) {
    float_contours.push_back(convertContourTo2f(c));
  }
  return float_contours;
}

} // namespace cv_process
