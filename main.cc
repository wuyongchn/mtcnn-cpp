#include <iostream>

#include "mtcnn.h"

int main() {
  ::google::InitGoogleLogging("");
  std::vector<std::string> model_file = {"../models/pnet.prototxt",
                                         "../models/rnet.prototxt",
                                         "../models/onet.prototxt"};
  std::vector<std::string> trained_file = {"../models/pnet.caffemodel",
                                           "../models/rnet.caffemodel",
                                           "../models/onet.caffemodel"};
  Params params;
  params.min_face_size = 40;

  MTCNN mtcnn(model_file, trained_file, params);

  std::string file = "/home/wuyong/CLionProjects/face_detection/test.jpg";
  cv::Mat img = cv::imread(file);
  std::vector<BBox> ans = mtcnn.Detect(img);

  for (auto& bbox : ans) {
    float x = bbox.tl.x;
    float y = bbox.tl.y;
    float h = bbox.br.y - bbox.tl.y + 1;
    float w = bbox.br.x - bbox.tl.x + 1;
    cv::rectangle(img, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);
    std::cout << x << ' ' << y << ' ' << h << ' ' << ' ' << w << std::endl;

    for (auto& landmark : bbox.landmarks) {
      cv::circle(img, cv::Point(landmark.x, landmark.y), 1,
                 cv::Scalar(255, 255, 0), 2);
      std::cout << landmark.x << ' ' << landmark.y << std::endl;
    }
  }
  cv::imshow("a", img);
  cv::waitKey(0);
  return 0;
}