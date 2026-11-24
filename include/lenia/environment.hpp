#pragma once

#include "lenia/convolve.hpp"
#include "lenia/step.hpp"

#include <cstddef>

namespace lenia {

// Spatial field E(x, y) in [-1, 1] that biases growth where it is positive and suppresses it
// where it is negative. With it the update becomes
//
//   next = clamp(current + dt * (G(U) + strength * E(x, y)), 0, 1)
//
// Coordinates are relative to the world center (dx = x - width / 2, dy = y - height / 2,
// integer division), the axis u = dx cos(angle) + dy sin(angle) points along `angle`, and
// `offset` moves the pattern along that axis (or, for Ripples, outward), so driving offset from
// time makes the environment drift. workbench_step.comp evaluates the same formulas
// (lenia_math.glsl, lenia_environment).
enum class EnvironmentPattern {
    Uniform = 0,  // 1 everywhere: a global growth bias
    Spot = 1,     // cos(pi * min(d / scale, 1)), d from the center moved by offset along angle
    Gradient = 2, // clamp((u - offset) / scale, -1, 1)
    Stripes = 3,  // sin(2 pi (u - offset) / scale)
    Ripples = 4,  // cos(2 pi (d - offset) / scale), d from the world center
};

struct Environment {
    EnvironmentPattern pattern = EnvironmentPattern::Uniform;
    float strength = 0.0f; // 0 disables the environment
    float scale = 64.0f;   // cells (wavelength, radius or ramp half-width); at least 1
    float angle = 0.0f;    // radians
    float offset = 0.0f;   // cells
};

float environment_field(const Environment& env, std::size_t x, std::size_t y, std::size_t width, std::size_t height);

// step_lenia with the environment term above. Equals step_lenia(grid, kernel, params) when
// env.strength == 0.
Grid step_lenia(const Grid& grid, const Grid& kernel, const GrowthParams& params, const Environment& env);

} // namespace lenia
