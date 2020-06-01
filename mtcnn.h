#ifndef FACE_DETECTION__MTCNN_H_
#define FACE_DETECTION__MTCNN_H_

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "bounding_box.h"
#include "caffe/caffe.hpp"

struct Params {
  float resize_factor = 0.709f;
  float confidence_thresh[3] = {0.6f, 0.7f, 0.7f};  // pnet, rnet, onet
  float nms_thresh1 = 0.5f;                         // intra-scale num threshold
  float nms_thresh = 0.7f;
  float mean_value = 127.5f;
  float normalize_factor = 0.0078125f;
  int min_face_size = 20;
  std::vector<std::string> offset_name = {"conv4-2", "conv5-2", "conv6-2"};
  std::string prob_name = "prob1";
  std::string landmark_name = "conv6-3";
  std::string input_name = "data";
  const int stage_num = 3;
  const int cell_size = 12;
  const int stride_size = 2;
  const int landmark_num = 5;
};

class MTCNN {
 public:
  MTCNN(const std::vector<std::string>& model_file,
        const std::vector<std::string>& trained_file, const Params& params);
  std::vector<BBox> Detect(const cv::Mat& img);
  float pnet_confidence_thresh() const { return params_.confidence_thresh[0]; }
  float rnet_confidence_thresh() const { return params_.confidence_thresh[1]; }
  float onet_confidence_thresh() const { return params_.confidence_thresh[2]; }
  void set_confidence_threshold(const float thresh, const int stage) {
    params_.confidence_thresh[stage - 1] = thresh;
  }
  float resize_factor() const { return params_.resize_factor; }
  void set_resize_factor(float resize_factor) {
    params_.resize_factor = resize_factor;
  }
  int min_face_size() const { return params_.min_face_size; }
  void set_min_face_size(int size) { params_.min_face_size = size; }

 private:
  void WrapInputLayer(int stage, int n, std::vector<cv::Mat>* input_channels);
  std::vector<float> PyramidImage(const cv::Size& img_size);
  void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
  void NetInference(const std::vector<cv::Mat>& imgs, const int stage);
  void PNetProcess(const cv::Mat& img, std::vector<BBox>& bboxes);
  void RNetProcess(const cv::Mat& img, std::vector<BBox>& bboxes);
  void ONetProcess(const cv::Mat& img, std::vector<BBox>& bboxes);

  std::vector<std::shared_ptr<caffe::Net<float>>> nets_;
  std::vector<cv::Size2i> input_geometry_;
  std::vector<int> input_channels_;
  Params params_;
};

#endif  // FACE_DETECTION__MTCNN_H_
