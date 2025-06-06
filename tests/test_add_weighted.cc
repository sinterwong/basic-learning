#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

class CVAddWeightedTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(CVAddWeightedTest, Normal) {
  // BGR: 一种蓝色
  cv::Mat original_image(300, 400, CV_8UC3, cv::Scalar(200, 50, 50));

  // 2. 创建一个示例单通道掩码 (例如，图像中心的一个白色矩形)
  cv::Mat mask_1channel = cv::Mat::zeros(original_image.size(), CV_8UC1);
  cv::Rect roi(100, 75, 200, 150); // x, y, width, height
  mask_1channel(roi).setTo(cv::Scalar(255));

  // 3. 定义要绘制的颜色 (例如，红色)
  cv::Scalar overlay_color(0, 0, 255); // BGR: 红色

  // 4. 准备 src2 (image_for_overlay)
  cv::Mat image_for_overlay = original_image.clone();
  image_for_overlay.setTo(overlay_color,
                          mask_1channel); // 在掩码区域设置为红色

  // 5. 定义权重和gamma
  double alpha = 1.0; // 原始图像的权重
  double beta = 0.5; // 叠加层图像的权重 (50% 透明度意味着叠加层贡献50%)
  double gamma = 0.0; // 偏移量

  // 6. 使用 cv::addWeighted 进行混合
  cv::Mat result_image;
  cv::addWeighted(original_image, alpha, image_for_overlay, beta, gamma,
                  result_image);

  // save results
  cv::imwrite("original_image.jpg", original_image);
  cv::imwrite("mask_1channel.jpg", mask_1channel);
  cv::imwrite("image_for_overlay.jpg", image_for_overlay);
  cv::imwrite("result_image.jpg", result_image);
}
