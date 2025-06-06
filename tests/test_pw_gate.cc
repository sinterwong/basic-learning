#include <fstream>
#include <string>
#include <vector>

#include <filesystem>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "pw_sampler.hpp"

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

  PWSampler pwSampler;
  PWGateInput input;
  input.seg_mask = mask;
  input.roi = roi;
  input.pleque_label = 255; // Use label 4 for plaque in this test data

  PWGateOutput output = pwSampler.getPWGate(input);

  // Visualize the result
  cv::Mat vis_img;
  cv::cvtColor(mask, vis_img, cv::COLOR_GRAY2BGR);

  // Draw the PW Gate
  cv::circle(vis_img, output.gate.center_point, 5, cv::Scalar(0, 255, 0), -1);
  // Draw the blood angle line (tangent)
  // Calculate endpoints for visualization
  float angle_rad = output.gate.blood_angle * CV_PI / 180.0f;
  cv::Point2f tangent_vec(std::cos(angle_rad), std::sin(angle_rad));
  cv::Point2f p1 = output.gate.center_point - tangent_vec * 50; // Extend line
  cv::Point2f p2 = output.gate.center_point + tangent_vec * 50;
  cv::line(vis_img, p1, p2, cv::Scalar(0, 255, 255), 2); // Yellow line

  // Draw the lumen diameter line (normal) if available
  if (output.gate.lumen_diameter > 0) {
    // The diameter endpoints are in ROI coordinates already
    cv::line(vis_img, output.debug_lumen_infos[0].diameter_points.first,
             output.debug_lumen_infos[0].diameter_points.second,
             cv::Scalar(255, 0, 0), 2); // Blue line
  }

  cv::imwrite((outputDir / "vis_pw_gate.png").string(), vis_img);

  // Assertions
  EXPECT_EQ(output.gate.scan_type,
            ScanType::Longitudinal); // Expect longitudinal for this data
  EXPECT_GT(output.gate.lumen_diameter, 0);
  EXPECT_GT(output.gate.center_point.x, 0);
  EXPECT_GT(output.gate.center_point.y, 0);
  EXPECT_NE(output.gate.blood_angle,
            0.0f); // Expect a non-zero angle for longitudinal

  // Check debug info if available
  if (input.debug) {
    EXPECT_FALSE(output.debug_lumen_infos.empty());
    for (const auto &info : output.debug_lumen_infos) {
      EXPECT_GT(info.diameter, 0);
      EXPECT_NE(info.diameter_points.first, info.diameter_points.second);
      // Check if tangent line is valid (not all zeros)
      EXPECT_FALSE(info.tangent_line.a == 0 && info.tangent_line.b == 0 &&
                   info.tangent_line.c == 0);
      // Check if tangent vector is not zero
      EXPECT_GT(cv::norm(info.tangent_vector), 0);
    }
  }
}
} // namespace testing_pw_gate