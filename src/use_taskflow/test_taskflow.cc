#include <opencv2/opencv.hpp>
#include <taskflow/taskflow.hpp>

#include <curl/curl.h>
#include <gflags/gflags.h>

DEFINE_string(image_root, "", "Specify the image root.");

static void readImages(std::string const &folder,
                       std::vector<cv::Mat> &images) {
  std::vector<cv::String> filenames;
  cv::glob(folder, filenames);
  for (const auto &filename : filenames) {
    cv::Mat image = cv::imread(filename);
    images.push_back(image);
  }
}

static void processFrame(const cv::Mat &input, cv::Mat &output) {
  cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(output, output, cv::Size(7, 7), 1.5, 1.5);
}

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("test taskflow");
  gflags::SetVersionString("0.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // 读取图像
  std::vector<cv::Mat> images;
  readImages(FLAGS_image_root, images);

  // 创建 Taskflow
  tf::Taskflow taskflow;

  // 创建任务流
  auto source = taskflow.emplace([&]() { std::cout << "Reading images...\n"; })
                    .name("source");

  std::vector<tf::Task> frames;
  for (size_t i = 0; i < images.size(); i++) {
    frames.push_back(taskflow
                         .emplace([&, i]() {
                           cv::Mat output;
                           processFrame(images[i], output);
                           std::stringstream filename;
                           filename << "output_" << i << ".jpg";
                           cv::imwrite(filename.str(), output);
                         })
                         .name("frame_" + std::to_string(i)));
  }

  auto sink = taskflow.emplace([&]() { std::cout << "Done.\n"; }).name("sink");

  // 创建数据依赖
  source.precede(frames.front());
  for (size_t i = 1; i < frames.size(); i++) {
    frames[i - 1].precede(frames[i]);
  }
  frames.back().precede(sink);

  // 执行任务流
  tf::Executor executor;
  executor.run(taskflow).wait();

  gflags::ShutDownCommandLineFlags();
  return 0;
}
