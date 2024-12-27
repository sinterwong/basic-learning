#include "contour_measure.hpp"

namespace cv_process {

std::pair<Contour, Contour>
ContourMeasure::process(const MeasureInput &seg_ret,
                        MeasureRet &measurement) const {
  return calcThicknessAndLength(seg_ret.contour_pairs, measurement);
}

double ContourMeasure::calculateDistance(cv::Point p1, cv::Point p2) const {
  return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

double ContourMeasure::dotProduct(cv::Point2f v1, cv::Point2f v2) const {
  return v1.x * v2.x + v1.y * v2.y;
}

double ContourMeasure::calculateExtension(
    const std::vector<cv::Point> &plaque_contour) const {
  if (plaque_contour.empty()) {
    return 0.0;
  }

  cv::Point minXY(INT_MAX, INT_MAX);
  cv::Point maxXY(INT_MIN, INT_MIN);

  for (const auto &point : plaque_contour) {
    minXY.x = std::min(minXY.x, point.x);
    minXY.y = std::min(minXY.y, point.y);
    maxXY.x = std::max(maxXY.x, point.x);
    maxXY.y = std::max(maxXY.y, point.y);
  }

  double ptpX = maxXY.x - minXY.x;
  double ptpY = maxXY.y - minXY.y;

  return std::max(ptpX, ptpY) * 2.0;
}

cv::Point2f ContourMeasure::calculateIntersection(cv::Point2f p1,
                                                  cv::Point2f p2,
                                                  cv::Point2f p3,
                                                  cv::Point2f p4) const {
  float x1 = p1.x, y1 = p1.y;
  float x2 = p2.x, y2 = p2.y;
  float x3 = p3.x, y3 = p3.y;
  float x4 = p4.x, y4 = p4.y;

  float denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

  if (denominator == 0) {
    return cv::Point2f(-1, -1);
  }

  float ua = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator;
  float ub = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denominator;

  if (0 <= ua && ua <= 1 && 0 <= ub && ub <= 1) {
    float x = x1 + ua * (x2 - x1);
    float y = y1 + ua * (y2 - y1);
    return cv::Point2f(x, y);
  }
  return cv::Point2f(-1, -1);
}

std::pair<Contour, Contour> ContourMeasure::calcThicknessAndLength(
    std::vector<std::pair<Contour, Contour>> const &contour_pairs,
    MeasureRet &measurement) const {
  // TODO: finish this code
  // init measurement
  measurement.thickness = 0;
  measurement.thichness_line = {};

  measurement.length = 0;
  measurement.length_line = {};

  Contour latest_plaque_contour;
  Contour latest_lumen_contour;

  for (auto const &[plaque_contour, lumen_contour] : contour_pairs) {
    int num_plaque_points = plaque_contour.size();
    std::vector<cv::Point2f> norm_vectors(num_plaque_points);
    for (int i = 0; i < num_plaque_points; i++) {
      cv::Point2f prev_point =
          plaque_contour[(i - 5 + num_plaque_points) % num_plaque_points];
      cv::Point2f next_point = plaque_contour[(i + 5) % num_plaque_points];
      cv::Point2f tangent_vectors = next_point - prev_point;
      norm_vectors[i] = cv::Point2f(-tangent_vectors.y, tangent_vectors.x);
    }

    // calculate distances and angles
    std::vector<std::vector<double>> distances(
        num_plaque_points, std::vector<double>(lumen_contour.size()));
    std::vector<std::vector<double>> angles(
        num_plaque_points, std::vector<double>(lumen_contour.size()));
    std::vector<std::vector<double>> scores(
        num_plaque_points, std::vector<double>(lumen_contour.size()));

    double median_dist = 0;
    std::vector<double> all_dists;
    for (int i = 0; i < num_plaque_points; i++) {
      for (int j = 0; j < lumen_contour.size(); j++) {
        all_dists.push_back(
            calculateDistance(plaque_contour[i], lumen_contour[j]));
      }
    }
    std::sort(all_dists.begin(), all_dists.end());
    if (!all_dists.empty()) {
      median_dist = all_dists[all_dists.size() / 2];
    }

    double global_min_distance = std::numeric_limits<double>::max();
    for (int i = 0; i < num_plaque_points; i++) {
      for (int j = 0; j < lumen_contour.size(); j++) {
        cv::Point2f diff_vec = plaque_contour[i] - lumen_contour[j];
        distances[i][j] =
            calculateDistance(plaque_contour[i], lumen_contour[j]);

        if (distances[i][j] < global_min_distance) {
          global_min_distance = distances[i][j];
        }

        double normal_norm = norm(norm_vectors[i]);
        double diff_norm = norm(diff_vec);

        cv::Point2f norm_vec_normalized = (normal_norm != 0)
                                              ? norm_vectors[i] / normal_norm
                                              : cv::Point2f(0, 0);

        cv::Point2f diff_vec_normalized =
            (diff_norm != 0) ? diff_vec / diff_norm : cv::Point2f(0, 0);

        double dp = dotProduct(norm_vec_normalized, diff_vec_normalized);
        angles[i][j] = acos(std::max(-1.0, std::min(1.0, dp)));

        double angle_thre = CV_PI / 4;
        double angle_score =
            (angles[i][j] / CV_PI) * angle_weight * median_dist;

        scores[i][j] = distances[i][j] + angle_score;

        if (distances[i][j] < dist_thre && angles[i][j] > angle_thre) {
          scores[i][j] = std::numeric_limits<double>::infinity();
        }
      }
    }

    if (global_min_distance > dist_thre) {
      continue;
    }

    // best match
    std::vector<int> best_matches(num_plaque_points);
    int max_thickness_idx = 0;
    double max_thickness = 0;
    for (int i = 0; i < num_plaque_points; i++) {
      auto minScoreIt = std::min_element(scores[i].begin(), scores[i].end());
      best_matches[i] = std::distance(scores[i].begin(), minScoreIt);
      if (distances[i][best_matches[i]] > max_thickness) {
        max_thickness = distances[i][best_matches[i]];
        max_thickness_idx = i;
      }
    }

    if (measurement.thickness < max_thickness) {
      measurement.thickness = max_thickness;
      measurement.thichness_line =
          std::make_pair(plaque_contour[max_thickness_idx],
                         lumen_contour[best_matches[max_thickness_idx]]);
      latest_plaque_contour = plaque_contour;
      latest_lumen_contour = lumen_contour;
    }
  }

  int num_plaque_points = latest_plaque_contour.size();
  cv::Point2f midpoint =
      (measurement.thichness_line.first + measurement.thichness_line.second) *
      0.5;
  cv::Point2f thickness_vec =
      measurement.thichness_line.second - measurement.thichness_line.first;
  cv::Point2f perpendicular_vec =
      cv::Point2f(-thickness_vec.y, thickness_vec.x);
  double perpendicular_norm = cv::norm(perpendicular_vec);
  perpendicular_vec = (perpendicular_norm != 0)
                          ? perpendicular_vec / perpendicular_norm
                          : cv::Point2f(0, 0);

  double extension = calculateExtension(latest_plaque_contour);

  cv::Point2f point1 = midpoint + perpendicular_vec * extension;
  cv::Point2f point2 = midpoint - perpendicular_vec * extension;

  std::vector<cv::Point2f> intersect_points;
  for (int i = 0; i < num_plaque_points; i++) {
    cv::Point2f p1 = latest_plaque_contour[i];
    cv::Point2f p2 = latest_plaque_contour[(i + 1) % num_plaque_points];

    cv::Point2f intersection = calculateIntersection(point1, point2, p1, p2);
    if (intersection.x != -1 && intersection.y != -1) {
      intersect_points.push_back(intersection);
    }
  }

  if (intersect_points.size() >= 2) {
    for (size_t i = 0; i < intersect_points.size(); i++) {
      for (size_t j = i + 1; j < intersect_points.size(); j++) {
        double dist =
            calculateDistance(intersect_points[i], intersect_points[j]);
        if (dist > measurement.length) {
          measurement.length = dist;
          measurement.length_line =
              std::make_pair(intersect_points[i], intersect_points[j]);
        }
      }
    }
  }

  return std::make_pair(latest_plaque_contour, latest_lumen_contour);
}

} // namespace cv_process