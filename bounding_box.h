#ifndef FACE_DETECTION__BOUNDING_BOX_H_
#define FACE_DETECTION__BOUNDING_BOX_H_
#include <vector>

struct Landmark {
  Landmark() = default;
  Landmark(const float x_, const float y_) : x(x_), y(y_) {}
  float x, y;
};

struct Offset {
  Offset() = default;
  Offset(const float dx1_, const float dy1_, const float dx2_, const float dy2_)
      : dx1(dx1_), dy1(dy1_), dx2(dx2_), dy2(dy2_) {}
  float dx1, dy1, dx2, dy2;
};

struct BBox {
  BBox() = default;
  BBox(const Landmark& tl, const Landmark& br, const float score);
  BBox(const Landmark& tl, const Landmark& br, const float score,
       const Offset& offset);
  BBox(const Landmark& tl, const Landmark& br, const float score,
       const std::vector<Landmark>& landmarks);
  BBox(const Landmark& tl, const Landmark& br, const float score,
       const Offset& offset, const std::vector<Landmark>& landmarks);
  BBox(const BBox& bbox);
  BBox& operator=(const BBox& bbox);
  float width() const { return br.x - tl.x + 1; }
  float height() const { return br.y - tl.y + 1; }
  float area() const;

  Landmark tl;  // top_left
  Landmark br;  // bottom_right
  float score;  // confidence score
  Offset offset;
  std::vector<Landmark> landmarks;
};

#endif  // FACE_DETECTION__BOUNDING_BOX_H_
