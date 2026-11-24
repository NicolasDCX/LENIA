#pragma once

#include "lenia/convolve.hpp"

namespace lenia {

// Growth mapping G(U), with z = (U - mu) / sigma. Both peak at G(mu) = 1 and fall to -1 far
// from mu. The values are passed to the shaders as is (lenia_math.glsl).
enum class GrowthFunction {
    Gaussian = 0,   // 2 * exp(-z^2 / 2) - 1                (Lenia "exponential" growth)
    Polynomial = 1, // 2 * max(0, 1 - z^2 / 9)^4 - 1        (Lenia "polynomial" growth, Orbium's)
};

struct GrowthParams {
    float mu = 0.15f;
    float sigma = 0.015f;
    float dt = 0.1f;
    GrowthFunction function = GrowthFunction::Gaussian;
    bool operator==(const GrowthParams&) const = default;
};

// G(u) for the given function, mu and sigma.
float growth(const GrowthParams& params, float u);

// Advances `grid` by one Lenia generation and returns the new grid; `grid` is not modified.
//
//   U      = convolve(grid, kernel)
//   next   = clamp(current + dt * growth(params, U), 0, 1)
Grid step_lenia(const Grid& grid, const Grid& kernel, const GrowthParams& params);

} // namespace lenia
