#ifndef FACE_DETECTION__UTILS_H_
#define FACE_DETECTION__UTILS_H_

#include <opencv2/opencv.hpp>
#include <vector>

#include "bounding_box.h"

namespace utils {
enum IouMethod { Union, Min };

float BBoxIoU(const BBox& bbox1, const BBox& bbox2, IouMethod method);
void BBoxNMS(std::vector<BBox>& bbox_vec, const float threshold,
             IouMethod method = Union);
std::vector<std::vector<BBox>> GenerateBBox(
    // feature map layout [B, *, H, W]
    const float* offset, const float* prob, const int batch, const int width,
    const int height, const int stride, const int cell_size, const float thresh,
    const float scale);
void BBox2Square(std::vector<BBox>& bbox_vec);
void BBoxRegress(std::vector<BBox>& bbox_vec, const float delta);
cv::Vec4i BBoxPadding(const BBox& bbox, const cv::Size& size, cv::Rect& rect);

}  // namespace utils

#endif  // FACE_DETECTION__UTILS_H_
