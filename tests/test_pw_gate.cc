#include <fstream>
#include <string>
#include <vector>

#include <filesystem>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "pw_locator.hpp"

namespace testing_pw_gate {
namespace fs = std::filesystem;
using namespace cv_process;

void collectFiles(const fs::path &directory,
                  std::vector<std::string> &resultPaths,
                  const std::vector<std::string> &extensions) {
  for (const auto &entry : fs::directory_iterator(directory)) {
    if (entry.is_regular_file()) {
      const std::string ext = entry.path().extension().string();
      for (const auto &validExt : extensions) {
        if (ext == validExt) {
          resultPaths.push_back(entry.path().string());
          break;
        }
      }
    }
  }
}

class PWGateTest : public ::testing::Test {
protected:
  void SetUp() override {
    // get all image paths from dir
    collectFiles(dataDir, maskPaths, {".png", ".jpg", ".jpeg"});

    // create vis output
    if (!fs::exists(outputDir)) {
      fs::create_directory(outputDir);
    }
  }
  void TearDown() override {}
  fs::path dataDir = fs::path("data/pw");
  fs::path maskDir = dataDir / "mask";
  fs::path outputDir = fs::path("vis_output");
  std::vector<std::string> maskPaths;
};

TEST_F(PWGateTest, Normal) {
  std::string maskPath = (maskDir / "anno_output.png").string();
  cv::Mat mask = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);

  mask = mask(cv::Rect{600, 100, 780, 700});
  cv::Rect roi{0, 0, mask.cols, mask.rows};

  cv::imwrite((outputDir / "vis_lbr_mask.png").string(), mask(roi));

  PWGateLocator pwLocator(mask, roi, 255, 0.3f, 2.f, 0.5f);

  PWGate gate = pwLocator.locate();

  // Print the result
  std::cout << "PW Gate Center: (" << gate.center_point.x << ", "
            << gate.center_point.y << ")" << std::endl;
  std::cout << "Blood Angle: " << gate.blood_angle << " degrees" << std::endl;
  std::cout << "Lumen Diameter: " << gate.lumen_diameter << std::endl;
  std::cout << "Scan Type: "
            << (gate.scan_type == ScanType::Longitudinal ? "Longitudinal"
                                                         : "Others")
            << std::endl;

  // Assertions (basic checks)
  // EXPECT_GT(gate.lumen_diameter, 0); // Diameter might be -1 if not found
  // EXPECT_NE(gate.center_point.x, 0); // Should be within ROI
  // EXPECT_NE(gate.center_point.y, 0); // Should be within ROI

  // Visualize the result
  cv::Mat vis_img;
  cv::cvtColor(mask, vis_img, cv::COLOR_GRAY2BGR);

  // Draw the PW gate center
  cv::circle(vis_img, gate.center_point, 5, cv::Scalar(0, 255, 0),
             -1); // Green circle

  // Draw the blood flow angle line (tangent)
  if (gate.scan_type == ScanType::Longitudinal && gate.lumen_diameter > 0) {
    // Calculate points on the tangent line for visualization
    double angle_rad = -gate.blood_angle * CV_PI / 180.0;
    cv::Point2f tangent_vec(std::cos(angle_rad), std::sin(angle_rad));
    cv::Point2f p1 = gate.center_point - tangent_vec * 50; // Extend line
    cv::Point2f p2 = gate.center_point + tangent_vec * 50; // Extend line
    cv::line(vis_img, p1, p2, cv::Scalar(255, 0, 0), 2);

    // Draw the diameter line (normal)
    cv::Point2f normal_vec(-tangent_vec.y,
                           tangent_vec.x); // Perpendicular vector
    cv::Point2f d1 =
        gate.center_point - normal_vec * (gate.lumen_diameter / 2.0);
    cv::Point2f d2 =
        gate.center_point + normal_vec * (gate.lumen_diameter / 2.0);
    cv::line(vis_img, d1, d2, cv::Scalar(0, 0, 255),
             2); // Red line for diameter
  }

  cv::imwrite((outputDir / "vis_pw_gate.png").string(), vis_img);
}
} // namespace testing_pw_gate