#pragma once

#include "lenia/convolve.hpp"

#include <cstddef>
#include <vector>

namespace lenia {

// Radial profile of one ring, evaluated at t in [0, 1] across the ring's width.
// All cores are 0 at t <= 0 and t >= 1 and peak at 1 (at t = 0.5 for the smooth ones).
enum class KernelCore {
    ExponentialBump, // exp(4 - 1 / (t * (1 - t)))         (Lenia "exponential")
    Polynomial,      // (4 * t * (1 - t))^4                 (Lenia "polynomial", Orbium O2u's core)
    Step,            // 1 for 1/4 <= t <= 3/4, else 0       (Lenia "step")
    GaussianBell,    // exp(-0.5 * ((t - 0.5) / 0.15)^2)
};

float kernel_core(KernelCore core, float t);

// Multi-ring Lenia kernel ("kernel shell"). For a cell at Euclidean distance d from the
// center, with r = d / radius and B = rings.size():
//
//   K(r) = rings[floor(B * r)] * core(B * r - floor(B * r))   for r < 1, else 0
//
// Ring weights are signed (negative rings inhibit) and listed innermost first.
struct KernelShape {
    int radius = 13;
    KernelCore core = KernelCore::ExponentialBump;
    std::vector<float> rings{1.0f};
};

// K(r) for r = distance / radius. Throws std::invalid_argument if rings is empty.
float kernel_shell(const KernelShape& shape, float r);

// (2 * radius + 1)^2 kernel of raw (unnormalized) K values; the middle cell is the origin.
// Throws std::invalid_argument if radius < 1 or rings is empty.
Grid make_kernel(const KernelShape& shape);

// Time-varying deformation of a KernelShape, driven by the node graph (node_graph.hpp). All
// fields are 0 when neutral, and a neutral warp reproduces make_kernel(shape) exactly.
//
//   effective radius  Re = radius * (1 + radius_delta), at least 1 cell
//   ring coordinate   t -> t^(2^peak_shift)            (> 0 moves each ring's peak outward)
//   core value        c -> c^(2^sharpness)             (> 0 narrows rings, < 0 widens them)
//
// These change the profile itself, not just its scale, so they survive normalization.
struct KernelWarp {
    float radius_delta = 0.0f;
    float peak_shift = 0.0f;
    float sharpness = 0.0f;

    bool neutral() const { return radius_delta == 0.0f && peak_shift == 0.0f && sharpness == 0.0f; }
    bool operator==(const KernelWarp&) const = default;
};

// Re for the warp, in cells.
double warped_radius(const KernelShape& shape, const KernelWarp& warp);

// K(r) of the warped shape, for r = distance / warped_radius.
float kernel_shell(const KernelShape& shape, const KernelWarp& warp, float r);

// (2 * ceil(Re) + 1)^2 kernel of raw warped K values. Same preconditions as make_kernel.
Grid make_kernel(const KernelShape& shape, const KernelWarp& warp);

struct KernelStats {
    double sum = 0.0;      // sum of all weights
    double positive = 0.0; // sum of positive weights
    double negative = 0.0; // sum of negative weights (<= 0)
    std::size_t nonzero = 0;
};

KernelStats kernel_stats(const Grid& kernel);

enum class KernelNormalization {
    None,        // raw weights
    Sum,         // divide by the sum: weights sum to 1 (Lenia's convention)
    Positive,    // divide by the positive sum: excitatory weights sum to 1, inhibitory keep their ratio
    AbsoluteSum, // divide by the sum of |w|: |weights| sum to 1
};

// Returns a scaled copy of `kernel`. Signs are never flipped: throws std::invalid_argument if
// the divisor the mode needs is not positive (e.g. Sum on a kernel whose weights cancel out).
Grid normalize_kernel(const Grid& kernel, KernelNormalization mode);

} // namespace lenia
