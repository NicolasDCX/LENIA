#include "lenia/convolve.hpp"

#include <stdexcept>

namespace lenia {

namespace {
// Wraps i into [0, n) for any signed i.
std::size_t wrap(long long i, std::size_t n) {
    const long long m = static_cast<long long>(n);
    return static_cast<std::size_t>(((i % m) + m) % m);
}
} // namespace

Grid convolve(const Grid& grid, const Grid& kernel) {
    if (kernel.width % 2 == 0 || kernel.height % 2 == 0) {
        throw std::invalid_argument("kernel dimensions must be odd");
    }

    Grid out(grid.width, grid.height, 0.0f);
    const long long cx = static_cast<long long>(kernel.width / 2);
    const long long cy = static_cast<long long>(kernel.height / 2);

    for (std::size_t y = 0; y < grid.height; ++y) {
        for (std::size_t x = 0; x < grid.width; ++x) {
            float sum = 0.0f;
            for (std::size_t ky = 0; ky < kernel.height; ++ky) {
                for (std::size_t kx = 0; kx < kernel.width; ++kx) {
                    const float w = kernel.at(kx, ky);
                    if (w == 0.0f) continue;
                    const long long dx = static_cast<long long>(kx) - cx;
                    const long long dy = static_cast<long long>(ky) - cy;
                    const std::size_t sx = wrap(static_cast<long long>(x) - dx, grid.width);
                    const std::size_t sy = wrap(static_cast<long long>(y) - dy, grid.height);
                    sum += w * grid.at(sx, sy);
                }
            }
            out.at(x, y) = sum;
        }
    }
    return out;
}

} // namespace lenia
