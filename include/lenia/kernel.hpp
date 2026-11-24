#pragma once

#include "lenia/convolve.hpp"

namespace lenia {

// Builds a square, odd-sized ring kernel for use with convolve / step_lenia.
//
// A cell at Euclidean distance d (in cells) from the center belongs to the ring when
//   radius - width / 2 <= d < radius + width / 2
// Ring cells get equal weight, all other cells are zero, and weights sum to 1.
// The kernel is the smallest odd size that contains every ring cell.
//
// Throws std::invalid_argument if radius or width is nonpositive or nonfinite,
// or if the ring contains no cell centers.
Grid make_ring_kernel(float radius, float width);

} // namespace lenia
