#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <typeinfo>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

enum class CalculateType { MEAN, MEDIAN, MAX, MODE };

double calculate_const(const std::vector<short> &nums, double exclude_range_l,
                       double exclude_range_r, CalculateType calculate_type) {
  if (nums.empty()) {
    throw std::invalid_argument("Input vector is empty");
  }

  std::vector<short> sorted_nums(nums.size());

  std::partial_sort_copy(nums.begin(), nums.end(), sorted_nums.begin(),
                         sorted_nums.end());

  size_t l_index = static_cast<size_t>(sorted_nums.size() * exclude_range_l);
  size_t r_index = sorted_nums.size() -
                   static_cast<size_t>(sorted_nums.size() * exclude_range_r);

  if (l_index >= r_index) {
    l_index = 0;
    r_index = sorted_nums.size();
  }

  // using iterator instead of copying a vector
  auto valid_begin = sorted_nums.begin() + l_index;
  auto valid_end = sorted_nums.begin() + r_index;
  size_t valid_size = r_index - l_index;

  if (valid_size == 0) {
    throw std::invalid_argument("No valid data after range exclusion");
  }

  switch (calculate_type) {
  case CalculateType::MEAN:
    return std::accumulate(valid_begin, valid_end, 0.0) / valid_size;
  case CalculateType::MEDIAN:
    return *(valid_begin + valid_size / 2);
  case CalculateType::MAX:
    return *(valid_end - 1);
  case CalculateType::MODE: {
    std::unordered_map<short, size_t> count_map;
    short mode = *valid_begin;
    size_t max_count = 0;
    for (auto it = valid_begin; it != valid_end; ++it) {
      size_t &count = ++count_map[*it];
      if (count > max_count) {
        mode = *it;
        max_count = count;
      }
    }
    return mode;
  }
  default:
    throw std::invalid_argument("Invalid calculate type");
  }
}

std::string roundToDecimalPlaces(double value, int decimalPlaces) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(decimalPlaces) << value; // 设置格式
  std::string roundedString = out.str(); // 获取格式化后的字符串

  // 将字符串转换回 double
  return roundedString;
}

std::map<std::string, std::map<std::string, fs::path>>
process_files(const fs::path &data_path) {
  // 构建各个路径
  fs::path images_path = data_path / "images";
  fs::path labels_path = data_path / "labels";
  fs::path height_maps_path = data_path / "height_maps";

  std::map<std::string, std::map<std::string, fs::path>> uu_infos;

  for (const auto &entry : fs::directory_iterator(labels_path)) {
    if (entry.is_regular_file()) {
      std::string filename = entry.path().filename().string();
      std::string uuid = filename.substr(0, filename.find_last_of('.'));

      size_t pos = filename.find(".txt");

      fs::path heigth_path =
          height_maps_path / filename.replace(filename.find(".txt"), 4, ".bin");
      fs::path image_path =
          images_path / filename.replace(pos, 4, "_white.png");

      uu_infos[uuid]["label_path"] =
          labels_path / entry.path().filename().string();
      uu_infos[uuid]["image_path"] = image_path;
      uu_infos[uuid]["height_path"] = heigth_path;
    }
  }
  return uu_infos;
}

// 将 std::map 保存到 CSV 文件
void save_to_csv(
    const std::map<std::string, std::map<std::string, std::string>> &data,
    const std::string &filename) {
  std::ofstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing.\n";
    return;
  }

  // 写入文件头
  file << "uuid, box_id, height_calculate_value\n";

  // 遍历外层 map
  for (const auto &outer_pair : data) {
    const std::string &outer_key = outer_pair.first;
    const auto &inner_map = outer_pair.second;

    // 遍历内层 map
    for (const auto &inner_pair : inner_map) {
      const std::string &inner_key = inner_pair.first;
      const std::string &value = inner_pair.second;

      // 写入 CSV 文件
      file << outer_key << ',' << inner_key << ',' << value << '\n';
    }
  }

  file.close();
}

int main() {
  std::string data_path =
      "/home/sinter/workspace/basic-learning/src/point_cloud/data/heatmap";
  fs::path data_f(data_path);

  auto uu_infos = process_files(data_f);

  std::map<std::string, std::map<std::string, std::string>>
      calculate_height_out_map;

  for (const auto &uu_info : uu_infos) {
    const std::string &uuid = uu_info.first;
    const auto &valued = uu_info.second;

    fs::path image_path = valued.at("image_path");
    fs::path label_path = valued.at("label_path");
    fs::path height_path = valued.at("height_path");

    if (!(fs::exists(image_path) && fs::exists(label_path) &&
          fs::exists(height_path))) {
      std::cerr << "One or more files do not exist" << std::endl;
      return -1;
    }

    //  读取二进制文件
    std::ifstream file(height_path, std::ios::binary);
    if (!file) {
      std::cerr << "无法打开文件: " << height_path << std::endl;
      return -1;
    }
    // 获取文件大小
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    cv::Mat image =
        cv::imread(image_path.string()); //.path().filename().string()

    // 获取图像尺寸
    int cols = image.cols; // 图像宽度
    int rows = image.rows; // 图像高度

    // 创建 height_map 矩阵
    std::vector<int16_t> buffer(rows * cols);
    file.read(reinterpret_cast<char *>(buffer.data()),
              buffer.size() * sizeof(int16_t));
    std::cout << "Element type: " << typeid(buffer[0]).name() << std::endl;

    file.close();

    auto result = std::minmax_element(buffer.begin(), buffer.end());
    // 将数据转换为 OpenCV 矩阵
    cv::Mat height_map(rows, cols, CV_16SC1, buffer.data());

    // FIXME: visualize
    cv::Mat height_map_normalized;
    cv::normalize(height_map, height_map_normalized, 0, 255, cv::NORM_MINMAX,
                  CV_8U);
    cv::imwrite("height_map_" + uuid + ".png", height_map_normalized);

    // 比较矩阵
    if (image.size() == height_map.size()) {
      std::cout << "Matrices are identical." << std::endl;
    } else {
      std::cout << "Matrices are different." << std::endl;
    }

    // 读取txt文件夹
    std::ifstream txt_file(label_path.string());

    // 检查文件是否成功打开
    if (!txt_file.is_open()) {
      std::cerr << "Failed to open the file!" << std::endl;
      return 1;
    }

    // 读取文件的每一行
    std::string line;
    std::string COPLANARITY_box_name = "BODY"; // GROUP_BODY_COPLANARITY
    std::string tips_box_name = "GROUP_PIN_TIP";
    std::string FILLET_box_name = "GROUP_SOLDER_FILLET";

    // 读取并忽略标题行
    std::getline(txt_file, line);

    // 处理数据行
    int box_id = 0;
    while (std::getline(txt_file, line)) {
      std::istringstream line_stream(line);

      // 解析每一行的数据
      std::string category;
      std::string difficult;

      int x1, y1, x2, y2, x3, y3, x4, y4;

      // 读取数据
      line_stream >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4 >> category >>
          difficult;

      if ((category == COPLANARITY_box_name) || (category == tips_box_name) ||
          (category == FILLET_box_name)) {
        // 定义多边形顶点
        std::vector<cv::Point> det_coords = {
            cv::Point(x1, y1), cv::Point(x2, y2), cv::Point(x3, y3),
            cv::Point(x4, y4)};
        cv::Mat mask = cv::Mat::zeros(height_map.size(), CV_8UC1);

        // 填充多边形区域为 1
        std::vector<std::vector<cv::Point>> polygons;
        polygons.push_back(det_coords);
        cv::fillPoly(mask, polygons, cv::Scalar(1));

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(mask, &minVal, &maxVal, &minLoc, &maxLoc);

        // FIXME: visualize
        cv::imwrite("mask_" + uuid + "_" + std::to_string(box_id) + ".png",
                    mask * 255);

        // 应用掩码到 height_map
        cv::Mat height_map_filtered =
            cv::Mat::zeros(height_map.size(), height_map.type());
        height_map.copyTo(height_map_filtered, mask);

        // FIXME: visualize
        cv::Mat masked_height_map_8U;
        cv::normalize(height_map_filtered, masked_height_map_8U, 0, 255,
                      cv::NORM_MINMAX, CV_8U);
        cv::imwrite("masked_height_map_" + uuid + "_" +
                        std::to_string(box_id++) + ".png",
                    masked_height_map_8U);

        // 获取掩码区域内的所有非零值
        std::vector<short> cur_draft_pcd;
        for (int i = 0; i < height_map_filtered.rows; ++i) {
          for (int j = 0; j < height_map_filtered.cols; ++j) {
            if (height_map_filtered.at<short>(i, j) != 0) {
              cur_draft_pcd.push_back(height_map_filtered.at<short>(i, j));
            }
          }
        }

        // CalculateType calculate_type = CalculateType:: MEAN;
        double box_get_value =
            calculate_const(cur_draft_pcd, 0, 0, CalculateType::MEDIAN);
        std::string box_get_value_string =
            roundToDecimalPlaces(box_get_value, 2);

        std::string pos_string = ' ' + category;
        std::string coords_line = line.substr(0, line.find(pos_string));

        std::replace(coords_line.begin(), coords_line.end(), ' ', '_');

        std::string box_id = coords_line + '_' + category;
        calculate_height_out_map[uuid][box_id] = box_get_value_string;
      }
    }
    // 关闭文件
    txt_file.close();
  }
  save_to_csv(calculate_height_out_map, "./ceshi_median.csv");
}