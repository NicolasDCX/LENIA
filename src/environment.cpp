#include "lenia/environment.hpp"

#include <algorithm>
#include <cmath>

namespace lenia {

float environment_field(const Environment& env, std::size_t x, std::size_t y, std::size_t width, std::size_t height) {
    constexpr float kPi = 3.14159265358979f;
    const float dx = static_cast<float>(x) - static_cast<float>(width / 2);
    const float dy = static_cast<float>(y) - static_cast<float>(height / 2);
    const float scale = std::max(env.scale, 1.0f);
    const float c = std::cos(env.angle);
    const float s = std::sin(env.angle);
    const float u = dx * c + dy * s - env.offset;
    switch (env.pattern) {
    case EnvironmentPattern::Uniform:
        return 1.0f;
    case EnvironmentPattern::Spot: {
        const float ex = dx - env.offset * c;
        const float ey = dy - env.offset * s;
        return std::cos(kPi * std::min(std::sqrt(ex * ex + ey * ey) / scale, 1.0f));
    }
    case EnvironmentPattern::Gradient:
        return std::clamp(u / scale, -1.0f, 1.0f);
    case EnvironmentPattern::Stripes:
        return std::sin(2.0f * kPi * u / scale);
    case EnvironmentPattern::Ripples:
        return std::cos(2.0f * kPi * (std::sqrt(dx * dx + dy * dy) - env.offset) / scale);
    }
    return 0.0f;
}

Grid step_lenia(const Grid& grid, const Grid& kernel, const GrowthParams& params, const Environment& env) {
    if (env.strength == 0.0f) return step_lenia(grid, kernel, params);
    Grid next = convolve(grid, kernel); // holds U, overwritten in place below
    for (std::size_t y = 0; y < next.height; ++y) {
        for (std::size_t x = 0; x < next.width; ++x) {
            const std::size_t i = y * next.width + x;
            const float g = growth(params, next.cells[i]) +
                            env.strength * environment_field(env, x, y, next.width, next.height);
            next.cells[i] = std::clamp(grid.cells[i] + params.dt * g, 0.0f, 1.0f);
        }
    }
    return next;
}

} // namespace lenia
