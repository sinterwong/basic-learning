/**
 * @file test_calc_fbank.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-11-21
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <gtest/gtest.h>

#include "fbank.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <sndfile.h>

#include <filesystem>

namespace fs = std::filesystem;

class LipSyncTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  fs::path dataDir = fs::path("data/lip_sync");

  fs::path audioPath = dataDir / "test.wav";

  fs::path imagePath = dataDir / "image.jpg";
};

std::vector<float> readAudioFile(std::string const &filePath) {
  SF_INFO sfinfo;
  SNDFILE *sndfile = sf_open(filePath.c_str(), SFM_READ, &sfinfo);
  if (!sndfile) {
    fprintf(stderr, "Error: could not open audio file: %s\n",
            sf_strerror(sndfile));
    return {};
  }

  int num_frames = sfinfo.frames;
  std::vector<float> audioData(num_frames);
  sf_readf_float(sndfile, audioData.data(), num_frames);
  sf_close(sndfile);
  return audioData;
}

std::vector<float> preprocessAudio(const std::vector<float> &audio) {
  int empty_frames_30 = 32 * 160; // 32 * 160 samples
  int empty_frames_31 = 35 * 160; // 35 * 160 samples

  std::vector<float> paddedAudio;
  paddedAudio.resize(empty_frames_30 + audio.size() + empty_frames_31);

  // Fill the beginning and end with zeros
  std::fill(paddedAudio.begin(), paddedAudio.begin() + empty_frames_30, 0.0f);
  std::transform(audio.begin(), audio.end(),
                 paddedAudio.begin() + empty_frames_30, [](float x) {
                   int16_t ret = static_cast<int16_t>(x * 32767.0f);
                   return static_cast<float>(ret);
                 });
  std::fill(paddedAudio.begin() + empty_frames_30 + audio.size(),
            paddedAudio.end(), 0.0f);

  return paddedAudio;
}

void visualize_fbank(const std::vector<std::vector<float>> &fbank_feature,
                     const std::string &output_path) {
  if (fbank_feature.empty() || fbank_feature[0].empty()) {
    std::cerr << "Empty feature matrix!" << std::endl;
    return;
  }

  // Get dimensions
  int num_frames = fbank_feature.size();
  int num_mel_bins = fbank_feature[0].size();

  // Find min and max values
  float max_val = fbank_feature[0][0];
  float min_val = fbank_feature[0][0];
  for (const auto &frame : fbank_feature) {
    for (float val : frame) {
      max_val = std::max(max_val, val);
      min_val = std::min(min_val, val);
    }
  }

  // Create image matrix
  cv::Mat image(num_mel_bins, num_frames, CV_8UC1);

  // Normalize and fill the image
  float scale = 255.0f / (max_val - min_val);
  for (int i = 0; i < num_frames; ++i) {
    for (int j = 0; j < num_mel_bins; ++j) {
      // Normalize to [0, 255] range
      float normalized_val = (fbank_feature[i][j] - min_val) * scale;
      // Flip Y axis (num_mel_bins - 1 - j)
      image.at<uchar>(num_mel_bins - 1 - j, i) =
          static_cast<uchar>(std::max(0.0f, std::min(255.0f, normalized_val)));
    }
  }

  // Apply color map
  cv::Mat color_image;
  cv::applyColorMap(image, color_image, cv::COLORMAP_JET);

  // Save image
  cv::imwrite(output_path, color_image);
}

TEST_F(LipSyncTest, TestFbankCalculation) {
  std::vector<float> audio = readAudioFile(audioPath.string());
  std::vector<float> preprocessedAudio = preprocessAudio(audio);
  ASSERT_EQ(preprocessedAudio.size(), 32 * 160 + audio.size() + 35 * 160);

  FbankComputer::FbankOptions opts;
  opts.num_mel_bins = 80;
  opts.frame_length = 25;
  opts.frame_shift = 10;
  opts.dither = 0.0;
  opts.energy_floor = 1.0;
  opts.sample_frequency = 16000;
  opts.use_log_fbank = true;
  opts.use_power = true;
  opts.window_type = "povey";

  FbankComputer fbankComputer(opts);
  auto fbankFeatures = fbankComputer.Compute(preprocessedAudio);
  ASSERT_GT(fbankFeatures.size(), 0);
  visualize_fbank(fbankFeatures, "fbank_features.png");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
