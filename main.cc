#include <iostream>

#include "mtcnn/mtcnn.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << argv[0] << " model_path image_path <min_size>" << std::endl;
    exit(0);
  }
  ::google::InitGoogleLogging(argv[0]);
  std::vector<std::string> model_file = {
      std::string(argv[1]) + "/pnet.prototxt",
      std::string(argv[1]) + "/rnet.prototxt",
      std::string(argv[1]) + "/onet.prototxt"};
  std::vector<std::string> trained_file = {
      std::string(argv[1]) + "/pnet.caffemodel",
      std::string(argv[1]) + "/rnet.caffemodel",
      std::string(argv[1]) + "/onet.caffemodel"};
  std::string file(argv[2]);
  Params params;
  if (argc == 4) {
    params.min_face_size = std::stoi(argv[3]);
  }

  MTCNN mtcnn(model_file, trained_file, params);
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
  cv::imshow("test", img);
  cv::waitKey(0);
  return 0;
}
