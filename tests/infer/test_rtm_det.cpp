#include "infer/infer_wrapper.hpp"
#include "infer/ncnn_frame_infer.hpp"
#include "infer_types.hpp"
#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>

using namespace infer;
using namespace infer::dnn;
class RTMDetInferenceTest : public ::testing::Test {
protected:
  void SetUp() override {
    rtmParam.name = "RTMDet";
    rtmParam.modelPath = "models/rtmdet.bin";
    rtmParam.paramPath = "models/rtmdet.param";
    rtmParam.deviceType = DeviceType::CPU;
    rtmParam.inputShape = {640, 640, 3};
    rtmParam.preprocessArg.meanVals = {127.5f, 127.5f, 127.5f};
    rtmParam.preprocessArg.normVals = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
  }
  void TearDown() override {}

  FrameInferParam rtmParam;
  std::string imagePath = "data/image.png";
};

TEST_F(RTMDetInferenceTest, TestInference) {
  ncnn_infer::FrameInference rtmInfer(rtmParam);
  ASSERT_EQ(rtmInfer.initialize(), InferErrorCode::SUCCESS);

  rtmInfer.prettyPrintModelInfos();

  AlgoInput input;
  cv::Mat image = cv::imread(imagePath);
  if (image.empty()) {
    FAIL() << "Could not read image";
  }
  input.setParams(FrameInput{image});

  ModelOutput output;
  ASSERT_EQ(rtmInfer.infer(input, output), InferErrorCode::SUCCESS);
  // auto *detRet = output.getParams<DetRet>();
  // ASSERT_NE(detRet, nullptr);
}

TEST_F(RTMDetInferenceTest, TestInferWrapper) {
  InferSafeWrapper<ncnn_infer::FrameInference, FrameInferParam> wrapper(
      rtmParam);
  ASSERT_EQ(wrapper.initialize(), InferErrorCode::SUCCESS);

  ASSERT_TRUE(wrapper.tryAcquire());
  Inference *infer = wrapper.get();
  ASSERT_NE(infer, nullptr);

  AlgoInput input;
  cv::Mat image = cv::imread("data/image.png");
  if (image.empty()) {
    FAIL() << "Could not read image";
  }
  input.setParams(FrameInput{image});

  ModelOutput output;
  ASSERT_EQ(infer->infer(input, output), InferErrorCode::SUCCESS);
  wrapper.release();
}
