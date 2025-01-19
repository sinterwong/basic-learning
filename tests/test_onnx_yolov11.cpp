#include "frame_infer.hpp"
#include "infer_wrapper.hpp"
#include "yoloDet.hpp"
#include "gtest/gtest.h"
#include <memory>
#include <opencv2/opencv.hpp>

using namespace infer::dnn;
using namespace infer;
class YoloInferenceTest : public ::testing::Test {
protected:
  void SetUp() override {
    YoloDetParams yoloParams;
    yoloParams.condThre = 0.5f;
    yoloParams.nmsThre = 0.5f;
    yoloParams.inputShape = {640, 640};
    params.setParams(yoloParams);

    inferParam.name = "test-yolov11";
    inferParam.modelPath = "models/yolov11n.onnx";
    inferParam.inputShape = yoloParams.inputShape;
    inferParam.deviceType = DeviceType::CPU;

    // infer engine
    engine = std::make_shared<FrameInference>(inferParam);

    // post-processor
    yoloDet = std::make_shared<vision::Yolov11Det>(params);
  }
  void TearDown() override {}

  std::string imagePath = "data/image.png";

  // infer engine
  FrameInferParam inferParam;
  std::shared_ptr<Inference> engine;

  // post-processor
  AlgoPostprocParams params;
  std::shared_ptr<vision::Vision> yoloDet;
};

TEST_F(YoloInferenceTest, TestInference) {

  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);
  engine->prettyPrintModelInfos();

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

  ASSERT_FALSE(image.empty());

  FrameInput frameInput;
  frameInput.image = imageRGB;
  frameInput.args.originShape = {imageRGB.cols, imageRGB.rows};
  frameInput.args.isEqualScale = true;
  frameInput.args.pad = {0, 0, 0};
  frameInput.args.meanVals = {0.0f, 0.0f, 0.0f};
  frameInput.args.normVals = {255.0f, 255.0f, 255.0f};

  AlgoInput algoInput;
  algoInput.setParams(frameInput);

  ModelOutput modelOutput;
  ASSERT_EQ(engine->infer(algoInput, modelOutput), InferErrorCode::SUCCESS);

  auto frameInputPtr = algoInput.getParams<FrameInput>();
  AlgoOutput algoOutput;
  ASSERT_TRUE(
      yoloDet->processOutput(modelOutput, frameInputPtr->args, algoOutput));

  auto *detRet = algoOutput.getParams<DetRet>();
  ASSERT_NE(detRet, nullptr);
  ASSERT_GT(detRet->bboxes.size(), 0);

  // visualize image
  cv::Mat visImage = image.clone();
  for (const auto &bbox : detRet->bboxes) {
    cv::rectangle(visImage, bbox.rect, cv::Scalar(0, 255, 0), 2);
    std::stringstream ss;
    ss << bbox.label << ":" << bbox.score;
    cv::putText(visImage, ss.str(), bbox.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 0, 255), 2);
  }
  cv::imwrite("vis.png", visImage);

  engine->terminate();
}

TEST_F(YoloInferenceTest, InferWrapperTest) {
  auto wrapper =
      std::make_shared<InferSafeWrapper<FrameInference, FrameInferParam>>(
          inferParam);
  ASSERT_NE(wrapper, nullptr);
  ASSERT_EQ(wrapper->initialize(), InferErrorCode::SUCCESS);

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

  ASSERT_FALSE(image.empty());

  FrameInput frameInput;
  frameInput.image = imageRGB;
  frameInput.args.originShape = {imageRGB.cols, imageRGB.rows};
  frameInput.args.roi = {100, 50, imageRGB.cols - 100, imageRGB.rows - 50};
  frameInput.args.isEqualScale = true;
  frameInput.args.pad = {0, 0, 0};
  frameInput.args.meanVals = {0.0f, 0.0f, 0.0f};
  frameInput.args.normVals = {255.0f, 255.0f, 255.0f};

  AlgoInput algoInput;
  algoInput.setParams(frameInput);

  ModelOutput modelOutput;
  ASSERT_TRUE(wrapper->tryAcquire());
  ASSERT_EQ(wrapper->get()->infer(algoInput, modelOutput),
            InferErrorCode::SUCCESS);
  wrapper->release();

  auto frameInputPtr = algoInput.getParams<FrameInput>();
  AlgoOutput algoOutput;
  ASSERT_TRUE(
      yoloDet->processOutput(modelOutput, frameInputPtr->args, algoOutput));

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
  cv::imwrite("vis_wrapper.png", visImage);

  wrapper->get()->terminate();
}