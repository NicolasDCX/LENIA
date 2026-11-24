#pragma once

#include <cstddef>
#include <vector>

namespace lenia {

// Row-major 2D grid of floats. Row 0 is the top row, column 0 is the left column.
struct Grid {
    std::size_t width = 0;
    std::size_t height = 0;
    std::vector<float> cells; // size == width * height

    Grid() = default;
    Grid(std::size_t w, std::size_t h, float fill = 0.0f)
        : width(w), height(h), cells(w * h, fill) {}

    float& at(std::size_t x, std::size_t y) { return cells[y * width + x]; }
    float at(std::size_t x, std::size_t y) const { return cells[y * width + x]; }
};

// Convolves `grid` with `kernel` using wraparound (toroidal) edges and raw weights.
//
// The kernel must have odd width and height. Its middle cell is the origin.
// Each input cell "stamps" the kernel onto the output at its own position:
// a single 1 at (x, y) produces a copy of the kernel centered at (x, y), in the
// same orientation (kernel's right is the grid's right, kernel's up is the grid's up).
//
//   out(x, y) = sum over (dx, dy) of kernel(dx, dy) * grid(x - dx, y - dy)
//
// Throws std::invalid_argument if the kernel has an even or zero dimension.
Grid convolve(const Grid& grid, const Grid& kernel);

} // namespace lenia
