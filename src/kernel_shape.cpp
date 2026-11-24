#include "lenia/kernel_shape.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lenia {

float kernel_core(KernelCore core, float t) {
    if (!(t > 0.0f && t < 1.0f)) return 0.0f; // also rejects NaN
    switch (core) {
    case KernelCore::ExponentialBump:
        return std::exp(4.0f - 1.0f / (t * (1.0f - t)));
    case KernelCore::Polynomial: {
        const float q = 4.0f * t * (1.0f - t);
        return (q * q) * (q * q);
    }
    case KernelCore::Step:
        return (t >= 0.25f && t <= 0.75f) ? 1.0f : 0.0f;
    case KernelCore::GaussianBell: {
        const float z = (t - 0.5f) / 0.15f;
        return std::exp(-0.5f * z * z);
    }
    }
    return 0.0f;
}

float kernel_shell(const KernelShape& shape, float r) {
    if (shape.rings.empty()) {
        throw std::invalid_argument("kernel shape needs at least one ring");
    }
    if (!(r >= 0.0f && r < 1.0f)) return 0.0f;
    const float br = static_cast<float>(shape.rings.size()) * r;
    const std::size_t ring = std::min(static_cast<std::size_t>(br), shape.rings.size() - 1);
    return shape.rings[ring] * kernel_core(shape.core, br - static_cast<float>(ring));
}

Grid make_kernel(const KernelShape& shape) {
    if (shape.radius < 1) {
        throw std::invalid_argument("kernel radius must be at least 1");
    }
    if (shape.rings.empty()) {
        throw std::invalid_argument("kernel shape needs at least one ring");
    }
    const long long half = shape.radius;
    const std::size_t size = static_cast<std::size_t>(2 * half + 1);
    Grid kernel(size, size, 0.0f);
    for (long long y = -half; y <= half; ++y) {
        for (long long x = -half; x <= half; ++x) {
            const float r = static_cast<float>(std::sqrt(static_cast<double>(x * x + y * y)) / shape.radius);
            kernel.at(static_cast<std::size_t>(x + half), static_cast<std::size_t>(y + half)) =
                kernel_shell(shape, r);
        }
    }
    return kernel;
}

double warped_radius(const KernelShape& shape, const KernelWarp& warp) {
    return std::max(1.0, shape.radius * (1.0 + static_cast<double>(warp.radius_delta)));
}

float kernel_shell(const KernelShape& shape, const KernelWarp& warp, float r) {
    if (shape.rings.empty()) {
        throw std::invalid_argument("kernel shape needs at least one ring");
    }
    if (!(r >= 0.0f && r < 1.0f)) return 0.0f;
    const float br = static_cast<float>(shape.rings.size()) * r;
    const std::size_t ring = std::min(static_cast<std::size_t>(br), shape.rings.size() - 1);
    float t = br - static_cast<float>(ring);
    if (warp.peak_shift != 0.0f) t = std::pow(t, std::exp2(warp.peak_shift));
    float c = kernel_core(shape.core, t);
    if (warp.sharpness != 0.0f && c > 0.0f) c = std::pow(c, std::exp2(warp.sharpness));
    return shape.rings[ring] * c;
}

Grid make_kernel(const KernelShape& shape, const KernelWarp& warp) {
    if (warp.neutral()) return make_kernel(shape);
    if (shape.radius < 1) {
        throw std::invalid_argument("kernel radius must be at least 1");
    }
    if (shape.rings.empty()) {
        throw std::invalid_argument("kernel shape needs at least one ring");
    }
    const double re = warped_radius(shape, warp);
    // Cells at distance >= Re are 0, so ceil(Re) covers everything nonzero.
    const long long half = static_cast<long long>(std::ceil(re - 1e-9));
    const std::size_t size = static_cast<std::size_t>(2 * half + 1);
    Grid kernel(size, size, 0.0f);
    for (long long y = -half; y <= half; ++y) {
        for (long long x = -half; x <= half; ++x) {
            const float r = static_cast<float>(std::sqrt(static_cast<double>(x * x + y * y)) / re);
            kernel.at(static_cast<std::size_t>(x + half), static_cast<std::size_t>(y + half)) =
                kernel_shell(shape, warp, r);
        }
    }
    return kernel;
}

Grid make_kernel(const KernelShape& shape, float shift_x, float shift_y) {
    if (shape.radius < 1) {
        throw std::invalid_argument("kernel radius must be at least 1");
    }
    if (shape.rings.empty()) {
        throw std::invalid_argument("kernel shape needs at least one ring");
    }
    if (!std::isfinite(shift_x) || !std::isfinite(shift_y)) {
        throw std::invalid_argument("kernel shift must be finite");
    }
    const double sx = shift_x, sy = shift_y;
    // Cells at distance >= radius from the ring center are 0.
    const long long half =
        static_cast<long long>(std::ceil(shape.radius + std::max(std::fabs(sx), std::fabs(sy)) - 1e-9));
    const std::size_t size = static_cast<std::size_t>(2 * half + 1);
    Grid kernel(size, size, 0.0f);
    for (long long y = -half; y <= half; ++y) {
        for (long long x = -half; x <= half; ++x) {
            // The ring center is at (-sx, -sy); unshifted this is make_kernel's distance exactly.
            const double dx = static_cast<double>(x) + sx;
            const double dy = static_cast<double>(y) + sy;
            const float r = static_cast<float>(std::sqrt(dx * dx + dy * dy) / shape.radius);
            kernel.at(static_cast<std::size_t>(x + half), static_cast<std::size_t>(y + half)) =
                kernel_shell(shape, r);
        }
    }
    return kernel;
}

KernelStats kernel_stats(const Grid& kernel) {
    KernelStats s;
    for (float w : kernel.cells) {
        s.sum += w;
        if (w > 0.0f) s.positive += w;
        if (w < 0.0f) s.negative += w;
        if (w != 0.0f) ++s.nonzero;
    }
    return s;
}

Grid normalize_kernel(const Grid& kernel, KernelNormalization mode) {
    const KernelStats s = kernel_stats(kernel);
    double divisor = 1.0;
    const char* what = nullptr;
    switch (mode) {
    case KernelNormalization::None:
        return kernel;
    case KernelNormalization::Sum:
        divisor = s.sum;
        what = "kernel weights must have a positive sum";
        break;
    case KernelNormalization::Positive:
        divisor = s.positive;
        what = "kernel needs at least one positive weight";
        break;
    case KernelNormalization::AbsoluteSum:
        divisor = s.positive - s.negative;
        what = "kernel needs at least one nonzero weight";
        break;
    }
    // Relative threshold: a sum that is only float cancellation noise counts as zero.
    const double scale = s.positive - s.negative;
    if (!(divisor > 1e-6 * scale) || !std::isfinite(divisor)) {
        throw std::invalid_argument(what);
    }

    Grid out = kernel;
    for (float& w : out.cells) {
        w = static_cast<float>(w / divisor);
    }
    return out;
}

} // namespace lenia
