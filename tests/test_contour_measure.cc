#include "contour_measure.hpp"
#include "seg_ret_postprocess.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <string>

namespace fs = std::filesystem;
using namespace cv_process;

class ContourMeasureTest : public ::testing::Test {
protected:
  void SetUp() override {
    // get all image paths from dir
    for (const auto &entry : fs::directory_iterator(dataDir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".png" ||
          entry.path().extension() == ".jpg" ||
          entry.path().extension() == ".jpeg") {
        maskPaths.push_back(entry.path().string());
      }
    }

    // create vis output
    if (!fs::exists(outputDir)) {
      fs::create_directory(outputDir);
    }
  }
  void TearDown() override {}
  fs::path dataDir = fs::path("data/measure/mask");
  fs::path outputDir = fs::path("vis_output");
  std::vector<std::string> maskPaths;

  SegPostProcessor segPostProcessor;
  ContourMeasure contourMeasure;
};

TEST_F(ContourMeasureTest, CalcThicknessAndLength) {
  for (const auto &maskPath : maskPaths) {
    cv::Mat mask = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);

    if (mask.empty()) {
      continue;
    }

    SegmentResult seg_ret;
    seg_ret.mask = mask;
    seg_ret.roi = cv::Rect{0, 0, mask.cols, mask.rows};

    MeasureInput measureInput = segPostProcessor.process(seg_ret);
    MeasureRet measureRet;
    contourMeasure.process(measureInput, measureRet);

    EXPECT_GT(measureRet.thickness, 0);
    EXPECT_GT(measureRet.length, 0);

    std::cout << "Thickness: " << measureRet.thickness << std::endl;
    std::cout << "Length: " << measureRet.length << std::endl;

    cv::Mat maskWithLines = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);
    cv::line(maskWithLines, measureRet.thichness_line.first,
             measureRet.thichness_line.second, cv::Scalar(128), 2);
    cv::line(maskWithLines, measureRet.length_line.first,
             measureRet.length_line.second, cv::Scalar(128), 2);
    std::string maskFileName = fs::path(maskPath).filename().string();
    std::string outputFileName = outputDir.string() + "/" + maskFileName;
    cv::imwrite(outputFileName, maskWithLines);
  }
}
