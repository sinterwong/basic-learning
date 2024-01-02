#include <iostream>
#include <opencv2/opencv.hpp>
#include <taskflow.hpp>
#include <vector>

using namespace cv;
using namespace std;

// 从文件夹中读取图像
void readImages(const string &folder, vector<Mat> &images) {
  vector<String> filenames;
  glob(folder, filenames);
  for (const auto &filename : filenames) {
    Mat image = imread(filename);
    images.push_back(image);
  }
}

// 处理视频帧并保存结果
void processFrame(const Mat &input, Mat &output) {
  cvtColor(input, output, COLOR_BGR2GRAY);
  GaussianBlur(output, output, Size(7, 7), 1.5, 1.5);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " <folder>\n";
    return 1;
  }

  // 读取图像
  vector<Mat> images;
  readImages(argv[1], images);

  // 创建 Taskflow
  tf::Taskflow taskflow;

  // 创建任务流
  auto source =
      taskflow.emplace([&]() { cout << "Reading images...\n"; }).name("source");

  vector<tf::Task> frames;
  for (size_t i = 0; i < images.size(); i++) {
    frames.push_back(taskflow
                         .emplace([&, i]() {
                           Mat output;
                           processFrame(images[i], output);
                           stringstream filename;
                           filename << "output_" << i << ".jpg";
                           imwrite(filename.str(), output);
                         })
                         .name("frame_" + to_string(i)));
  }

  auto sink = taskflow.emplace([&]() { cout << "Done.\n"; }).name("sink");

  // 创建数据依赖
  source.precede(frames.front());
  for (size_t i = 1; i < frames.size(); i++) {
    frames[i - 1].precede(frames[i]);
  }
  frames.back().precede(sink);

  // 执行任务流
  tf::Executor executor;
  executor.run(taskflow).wait();

  return 0;
}