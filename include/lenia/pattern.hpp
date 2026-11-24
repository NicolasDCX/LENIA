#pragma once

#include "lenia/convolve.hpp"
#include "lenia/kernel_shape.hpp"
#include "lenia/step.hpp"

#include <istream>
#include <string>

namespace lenia {

// A Lenia creature: initial cells plus the kernel and growth it was found with, and where it
// came from. Patterns are loaded from files rather than compiled in so their provenance stays
// with the data.
struct Pattern {
    std::string name;
    std::string source;  // citation or URL of the original data
    std::string license; // license the data is distributed under
    KernelShape kernel;
    GrowthParams growth;
    Grid cells;
};

// Parses a text pattern. Lines are "key value", '#' starts a comment, keys are:
//
//   name <text>          source <text>         license <text>
//   R <int>              rings <w0> <w1> ...   core exponential|polynomial|step|gaussian
//   mu <float>           sigma <float>         growth gaussian|polynomial
//   dt <float>           (or T <float>, meaning dt = 1 / T)
//   cells                (must be last; everything after it is the cell array)
//
// The cell array is rows of numbers, top row first. '[', ']' and ',' are accepted, so a
// Python / JSON nested list can be pasted as is: with brackets each ']' ends a row, without
// them each line is a row. Short rows are padded with zeros.
//
// Missing keys keep the KernelShape / GrowthParams defaults (so growth is Gaussian unless the
// file says otherwise). Throws std::invalid_argument on unknown keys or names, malformed
// numbers, or an empty cell array.
Pattern parse_pattern(std::istream& in);

// Opens `path` and parses it. Throws std::runtime_error if the file cannot be read.
Pattern load_pattern_file(const std::string& path);

// Copies `cells` into `world` centered on (cx, cy), wrapping at the edges and overwriting.
void stamp_pattern(Grid& world, const Grid& cells, std::size_t cx, std::size_t cy);

} // namespace lenia
