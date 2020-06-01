#include "utils.h"

#include <algorithm>

namespace utils {
  
float BBoxIoU(const BBox& bbox1, const BBox& bbox2, IouMethod method) {
  float inner_tlx = std::max<float>(bbox1.tl.x, bbox2.tl.x);
  float inner_tly = std::max<float>(bbox1.tl.y, bbox2.tl.y);
  float inner_brx = std::min<float>(bbox1.br.x, bbox2.br.x);
  float inner_bry = std::min<float>(bbox1.br.y, bbox2.br.y);
  float overlap_w = std::max<float>(inner_brx - inner_tlx + 1, 0);
  float overlap_h = std::max<float>(inner_bry - inner_tly + 1, 0);
  float intersection = overlap_w * overlap_h, denominator;
  if (method == Min) {
    denominator = std::min<float>(bbox1.area(), bbox2.area()) + 1e-7f;
  } else {
    denominator = bbox1.area() + bbox2.area() - intersection + 1e-7f;
  }
  return intersection / denominator;
}

void BBoxNMS(std::vector<BBox>& bbox_vec, const float threshold,
             IouMethod method) {
  if (bbox_vec.empty()) {
    return;
  }
  std::sort(bbox_vec.begin(), bbox_vec.end(),
            [](const BBox& bbox1, const BBox& bbox2) {
              return bbox1.score > bbox2.score;
            });
  std::vector<BBox> output;
  std::vector<bool> merged(bbox_vec.size(), false);
  for (int i = 0; i < bbox_vec.size(); ++i) {
    if (merged[i]) {
      continue;
    }
    output.push_back(bbox_vec[i]);
    for (int j = i + 1; j < bbox_vec.size(); ++j) {
      if (merged[j]) {
        continue;
      }
      if (BBoxIoU(bbox_vec[i], bbox_vec[j], method) > threshold) {
        merged[j] = true;
      }
    }
  }
  std::swap(bbox_vec, output);
}

std::vector<std::vector<BBox>> GenerateBBox(
    const float* offset, const float* prob, const int batch, const int width,
    const int height, const int stride, const int cell_size, const float thresh,
    const float scale) {
  int count = width * height;
  std::vector<std::vector<BBox>> candidates;
  for (int n = 0; n < batch; ++n) {
    std::vector<BBox> tmp;
    const float* confidence = prob + count;
    for (int i = 0; i < count; ++i) {
      if (confidence[i] >= thresh) {
        int idx_h = i / width;
        int idx_w = i - width * idx_h;
        BBox bbox;
        bbox.tl.x = std::ceil((idx_w * stride + 1) / scale);
        bbox.tl.y = std::ceil(idx_h * stride + 1) / scale;
        bbox.br.x = std::ceil(idx_w * stride + cell_size - 1 + 1) / scale;
        bbox.br.y = std::ceil(idx_h * stride + cell_size - 1 + 1) / scale;
        bbox.score = confidence[i];
        bbox.offset.dx1 = offset[i + 0 * count];
        bbox.offset.dy1 = offset[i + 1 * count];
        bbox.offset.dx2 = offset[i + 2 * count];
        bbox.offset.dy2 = offset[i + 3 * count];
        tmp.push_back(bbox);
      }
    }
    candidates.push_back(std::move(tmp));
    offset += 4 * count;
    prob += count;
  }
  return candidates;
}

void BBox2Square(std::vector<BBox>& bbox_vec) {
  for (auto& bbox : bbox_vec) {
    float width = bbox.width();
    float height = bbox.height();
    float side = std::max<float>(width, height);
    bbox.tl.x = bbox.tl.x + (width - side) * 0.5f;
    bbox.tl.y = bbox.tl.y + (height - side) * 0.5f;
    bbox.br.x = bbox.tl.x + side - 1;
    bbox.br.y = bbox.tl.y + side - 1;
  }
}

void BBoxRegress(std::vector<BBox>& bbox_vec, const float delta) {
  for (auto& bbox : bbox_vec) {
    float width = bbox.width() + delta;
    float height = bbox.height() + delta;
    bbox.tl.x = bbox.tl.x + width * bbox.offset.dx1;
    bbox.tl.y = bbox.tl.y + height * bbox.offset.dy1;
    bbox.br.x = bbox.br.x + width * bbox.offset.dx2;
    bbox.br.y = bbox.br.y + height * bbox.offset.dy2;
  }
}

cv::Vec4i BBoxPadding(const BBox& bbox, const cv::Size& size, cv::Rect& rect) {
  float xmin = std::max<float>(bbox.tl.x, 0.0f);
  float ymin = std::max<float>(bbox.tl.y, 0.0f);
  float xmax = std::min<float>(bbox.br.x, size.width - 1);
  float ymax = std::min<float>(bbox.br.y, size.height - 1);

  rect.x = std::ceil(xmin);
  rect.y = std::ceil(ymin);
  rect.width = std::ceil(xmax) - rect.x + 1;
  rect.height = std::ceil(ymax) - rect.y + 1;

  // left, right, top, bottom
  cv::Vec4i padding;
  padding[0] = std::abs(bbox.tl.x - xmin);  // left
  padding[1] = std::abs(bbox.br.x - xmax);  // right
  padding[2] = std::abs(bbox.tl.y - ymin);  // top
  padding[3] = std::abs(bbox.br.y - ymax);  // bottom
  return padding;
}

}  // namespace utils