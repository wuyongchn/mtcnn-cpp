#include "bounding_box.h"

BBox::BBox(const Landmark& tl0, const Landmark& br0, const float score0)
    : tl(tl0), br(br0), score(score0), offset(), landmarks() {}

BBox::BBox(const Landmark& tl0, const Landmark& br0, const float score0,
           const Offset& offset0)
    : tl(tl0), br(br0), score(score0), offset(offset0), landmarks() {}

BBox::BBox(const Landmark& tl0, const Landmark& br0, const float score0,
           const std::vector<Landmark>& landmarks0)
    : tl(tl0), br(br0), score(score0), offset(), landmarks(landmarks0) {}

BBox::BBox(const Landmark& tl0, const Landmark& br0, const float score0,
           const Offset& offset0, const std::vector<Landmark>& landmarks0)
    : tl(tl0), br(br0), score(score0), offset(offset0), landmarks(landmarks0) {}
BBox::BBox(const BBox& bbox)
    : tl(bbox.tl),
      br(bbox.br),
      score(bbox.score),
      offset(bbox.offset),
      landmarks(bbox.landmarks) {}
BBox& BBox::operator=(const BBox& bbox) {
  tl = bbox.tl;
  br = bbox.br;
  score = bbox.score;
  offset = bbox.offset;
  landmarks = bbox.landmarks;
  return *this;
}

float BBox::area() const {
  float w = std::max<float>(br.x - tl.x + 1, 0);
  float h = std::max<float>(br.y - tl.y + 1, 0);
  return w * h;
}