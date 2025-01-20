#include "infer/frame_infer.hpp"
#include "infer_types.hpp"
#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>

using namespace infer;
using namespace infer::dnn;
class RTMDetInferenceTest : public ::testing::Test {
protected:
  void SetUp() override {
    rtmParam.name = "test-rtmdet";
    rtmParam.modelPath = "models/rtmdet";
    rtmParam.inputShape = {640, 640};
    rtmParam.deviceType = DeviceType::CPU;
    engine = std::make_shared<FrameInference>(rtmParam);

    // TODO: init rtm postprocess
  }
  void TearDown() override {}
  std::string imagePath = "data/image.png";

  FrameInferParam rtmParam;
  std::shared_ptr<Inference> engine;

  // TODO: implement rtm post-processor
};

TEST_F(RTMDetInferenceTest, TestInference) {
  auto engine = std::make_shared<FrameInference>(rtmParam);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

  ASSERT_FALSE(image.empty());

  FrameInput frameInput;
  frameInput.image = imageRGB;
  frameInput.args.originShape = {imageRGB.cols, imageRGB.rows};
  frameInput.args.roi = {0, 0, imageRGB.cols, imageRGB.rows};
  frameInput.args.isEqualScale = true;
  frameInput.args.pad = {0, 0, 0};
  frameInput.args.meanVals = {0.0f, 0.0f, 0.0f};
  frameInput.args.normVals = {255.0f, 255.0f, 255.0f};

  AlgoInput algoInput;
  algoInput.setParams(frameInput);

  ModelOutput modelOutput;
  ASSERT_EQ(engine->infer(algoInput, modelOutput), InferErrorCode::SUCCESS);

  // TODO: add post-processor

  engine->terminate();
}
