#include "infer/infer_wrapper.hpp"
#include "infer/rtm_det.hpp"
#include "infer_types.hpp"
#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>

class RTMDetInferenceTest : public ::testing::Test {
protected:
  void SetUp() override {
    rtmDetParam.name = "RTMDet";
    rtmDetParam.modelPath = "models/rtmdet.bin";
    rtmDetParam.paramPath = "models/rtmdet.param";
    rtmDetParam.deviceType = infer::DeviceType::CPU;
    rtmDetParam.inputShape = {640, 640, 3};
    rtmDetParam.preprocessArg.meanVals = {127.5f, 127.5f, 127.5f};
    rtmDetParam.preprocessArg.normVals = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
  }
  void TearDown() override {}

  infer::RTMDetParam rtmDetParam;
  std::string imagePath = "data/image.png";
};

TEST_F(RTMDetInferenceTest, TestInference) {
  infer::dnn::RTMDetInference rtmInfer(rtmDetParam);
  ASSERT_EQ(rtmInfer.initialize(), infer::InferErrorCode::SUCCESS);

  rtmInfer.prettyPrintModelInfos();

  infer::AlgoInput input;
  cv::Mat image = cv::imread(imagePath);
  if (image.empty()) {
    FAIL() << "Could not read image";
  }
  input.setParams(infer::FrameInput{image});

  infer::AlgoOutput output;
  ASSERT_EQ(rtmInfer.infer(input, output), infer::InferErrorCode::SUCCESS);
  auto *detRet = output.getParams<infer::DetRet>();
  ASSERT_NE(detRet, nullptr);
}

TEST(RTMDetInferenceTest, TestInferWrapper) {
  infer::InferSafeWrapper<infer::dnn::RTMDetInference, infer::RTMDetParam>
      wrapper(infer::RTMDetParam{});
  ASSERT_EQ(wrapper.initialize(), infer::InferErrorCode::SUCCESS);

  ASSERT_TRUE(wrapper.tryAcquire());
  infer::dnn::AlgoInference *infer = wrapper.get();
  ASSERT_NE(infer, nullptr);

  infer::AlgoInput input;
  cv::Mat image = cv::imread("data/image.png");
  if (image.empty()) {
    FAIL() << "Could not read image";
  }
  input.setParams(infer::FrameInput{image});

  infer::AlgoOutput output;
  ASSERT_EQ(infer->infer(input, output), infer::InferErrorCode::SUCCESS);
  wrapper.release();
}