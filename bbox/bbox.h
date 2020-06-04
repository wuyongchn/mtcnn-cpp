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

#ifndef MTCNN_CPP_BBOX_BOX_H_
#define MTCNN_CPP_BBOX_BOX_H_
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

#endif  // MTCNN_CPP_BOUNDING_BOX_H_
