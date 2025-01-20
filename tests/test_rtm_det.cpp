#include "infer/frame_infer.hpp"
#include "infer_types.hpp"
#include "rtmDet.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

using namespace infer;
using namespace infer::dnn;
class RTMDetInferenceTest : public ::testing::Test {
protected:
  void SetUp() override {

    AnchorDetParams anchorDetParams;
    anchorDetParams.condThre = 0.5f;
    anchorDetParams.nmsThre = 0.45f;
    anchorDetParams.inputShape = {640, 640};
    params.setParams(anchorDetParams);

    rtmDet = std::make_shared<vision::RTMDet>(params);
    ASSERT_NE(rtmDet, nullptr);
  }
  void TearDown() override {}

  fs::path dataDir = fs::path("data");

  std::string imagePath = (dataDir / "thy_image.png").string();

  AlgoPostprocParams params;

  std::shared_ptr<vision::Vision> rtmDet;
};

TEST_F(RTMDetInferenceTest, NCNNImageInfer) {
  FrameInferParam rtmParam;
  rtmParam.name = "test-rtmdet";
  rtmParam.modelPath = "models/rtmdet.ncnn";
  rtmParam.inputShape = {640, 640};
  rtmParam.deviceType = DeviceType::CPU;

  std::shared_ptr<Inference> engine =
      std::make_shared<FrameInference>(rtmParam);
  ASSERT_NE(engine, nullptr);

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  FrameInput frameInput;
  frameInput.image = imageRGB;
  frameInput.args.originShape = {imageRGB.cols, imageRGB.rows};
  frameInput.args.roi = {0, 0, imageRGB.cols, imageRGB.rows};
  frameInput.args.isEqualScale = true;
  frameInput.args.pad = {0, 0, 0};
  frameInput.args.meanVals = {120.0f, 120.0f, 120.0f};
  frameInput.args.normVals = {60.0f, 60.0f, 60.0f};

  AlgoInput algoInput;
  algoInput.setParams(frameInput);

  ModelOutput modelOutput;
  ASSERT_EQ(engine->infer(algoInput, modelOutput), InferErrorCode::SUCCESS);

  auto frameInputPtr = algoInput.getParams<FrameInput>();
  AlgoOutput algoOutput;
  ASSERT_TRUE(
      rtmDet->processOutput(modelOutput, frameInputPtr->args, algoOutput));

  auto *detRet = algoOutput.getParams<DetRet>();
  ASSERT_NE(detRet, nullptr);
  ASSERT_GT(detRet->bboxes.size(), 0);

  cv::Mat visImage = image.clone();
  for (const auto &bbox : detRet->bboxes) {
    cv::rectangle(visImage, bbox.rect, cv::Scalar(0, 255, 0), 2);
    std::stringstream ss;
    ss << bbox.label << ":" << bbox.score;
    cv::putText(visImage, ss.str(), bbox.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 0, 255), 2);
  }
  cv::imwrite("vis_ncnn_rtmdet.png", visImage);

  engine->terminate();
}

TEST_F(RTMDetInferenceTest, ONNXImageInfer) {
  FrameInferParam rtmParam;
  rtmParam.name = "test-rtmdet";
  rtmParam.modelPath = "models/rtmdet.onnx";
  rtmParam.inputShape = {640, 640};
  rtmParam.deviceType = DeviceType::CPU;

  std::shared_ptr<Inference> engine =
      std::make_shared<FrameInference>(rtmParam);
  ASSERT_NE(engine, nullptr);

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  FrameInput frameInput;
  frameInput.image = imageRGB;
  frameInput.args.originShape = {imageRGB.cols, imageRGB.rows};
  frameInput.args.roi = {0, 0, imageRGB.cols, imageRGB.rows};
  frameInput.args.isEqualScale = true;
  frameInput.args.pad = {0, 0, 0};
  frameInput.args.meanVals = {120.0f, 120.0f, 120.0f};
  frameInput.args.normVals = {60.0f, 60.0f, 60.0f};

  AlgoInput algoInput;
  algoInput.setParams(frameInput);

  ModelOutput modelOutput;
  ASSERT_EQ(engine->infer(algoInput, modelOutput), InferErrorCode::SUCCESS);

  auto frameInputPtr = algoInput.getParams<FrameInput>();
  AlgoOutput algoOutput;
  ASSERT_TRUE(
      rtmDet->processOutput(modelOutput, frameInputPtr->args, algoOutput));

  auto *detRet = algoOutput.getParams<DetRet>();
  ASSERT_NE(detRet, nullptr);
  ASSERT_GT(detRet->bboxes.size(), 0);

  cv::Mat visImage = image.clone();
  for (const auto &bbox : detRet->bboxes) {
    cv::rectangle(visImage, bbox.rect, cv::Scalar(0, 255, 0), 2);
    std::stringstream ss;
    ss << bbox.label << ":" << bbox.score;
    cv::putText(visImage, ss.str(), bbox.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 0, 255), 2);
  }
  cv::imwrite("vis_onnx_rtmdet.png", visImage);

  engine->terminate();
}