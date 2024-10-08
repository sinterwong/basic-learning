#include "matrix.hpp"
#include <cmath>
#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace linear_algebra;

DEFINE_string(image_path, "", "Specify the image path.");

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("test taskflow");
  gflags::SetVersionString("0.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  cv::Mat image = cv::imread(FLAGS_image_path);

  if (image.empty()) {
    std::cerr << "Could not read the image." << std::endl;
    return 1;
  }

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
  gflags::ShutDownCommandLineFlags();
  return 0;
}
