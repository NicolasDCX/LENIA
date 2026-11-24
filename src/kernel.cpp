#include "lenia/kernel.hpp"

#include <cmath>
#include <stdexcept>

namespace lenia {

Grid make_ring_kernel(float radius, float width) {
    if (!std::isfinite(radius) || !std::isfinite(width) || radius <= 0.0f || width <= 0.0f) {
        throw std::invalid_argument("radius and width must be positive and finite");
    }

    const double inner = static_cast<double>(radius) - static_cast<double>(width) / 2.0;
    const double outer = static_cast<double>(radius) + static_cast<double>(width) / 2.0;

    // Ring cells satisfy |offset| <= d < outer, so the largest offset is ceil(outer) - 1.
    const long long half = static_cast<long long>(std::ceil(outer)) - 1;
    const std::size_t size = static_cast<std::size_t>(2 * half + 1);

    Grid kernel(size, size, 0.0f);
    std::size_t count = 0;
    for (long long y = -half; y <= half; ++y) {
        for (long long x = -half; x <= half; ++x) {
            const double d = std::sqrt(static_cast<double>(x * x + y * y));
            if (d >= inner && d < outer) {
                kernel.at(static_cast<std::size_t>(x + half), static_cast<std::size_t>(y + half)) = 1.0f;
                ++count;
            }
        }
    }

    if (count == 0) {
        throw std::invalid_argument("ring contains no cell centers");
    }

    const float weight = 1.0f / static_cast<float>(count);
    for (float& w : kernel.cells) {
        w *= weight;
    }
    return kernel;
}

} // namespace lenia
