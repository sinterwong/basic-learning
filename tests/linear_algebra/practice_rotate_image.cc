#include "matrix.hpp"
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace linear_algebra;
namespace fs = std::filesystem;

TEST(PracticeRotatedImageTest, Normal) {

  fs::path imagePath = fs::path("data/test_image.jpg");

  cv::Mat image = cv::imread(imagePath.string());

  ASSERT_FALSE(image.empty());

  cv::medianBlur(image, image, 5); // for link imgproc...

  // rotate 30 degree
  auto theta = M_PI / 6;
  Matrix<double> T{{{std::cos(theta), std::sin(theta)},
                    {-std::sin(theta), std::cos(theta)}}};

  cv::Mat rotatedImage = cv::Mat::zeros(image.rows, image.cols, image.type());
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      Vector<double> point({static_cast<double>(i), static_cast<double>(j)});
      auto new_point = T.dot(point);
      int new_i = static_cast<int>(new_point[0]);
      int new_j = static_cast<int>(new_point[1]);
      if (new_i >= 0 && new_i < image.rows && new_j >= 0 &&
          new_j < image.cols) {
        rotatedImage.at<cv::Vec3b>(new_i, new_j) = image.at<cv::Vec3b>(i, j);
      }
    }
  }
  cv::imwrite("rotated.jpg", rotatedImage);
}
