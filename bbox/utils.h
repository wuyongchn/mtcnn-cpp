/*  Copyright (C) <2020>  <Yong WU>
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MTCNN_CPP_BBOX_UTILS_H_
#define MTCNN_CPP_BBOX_UTILS_H_

#include <opencv2/opencv.hpp>
#include <vector>

#include "bbox/bbox.h"

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

#endif  // MTCNN_CPP_UTILS_H_
