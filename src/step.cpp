#include "lenia/step.hpp"

#include <algorithm>
#include <cmath>

namespace lenia {

float growth(const GrowthParams& params, float u) {
    const float z = (u - params.mu) / params.sigma;
    switch (params.function) {
    case GrowthFunction::Gaussian:
        break;
    case GrowthFunction::Polynomial: {
        const float q = std::max(0.0f, 1.0f - z * z / 9.0f);
        return 2.0f * (q * q) * (q * q) - 1.0f;
    }
    }
    return 2.0f * std::exp(-0.5f * z * z) - 1.0f;
}

Grid step_lenia(const Grid& grid, const Grid& kernel, const GrowthParams& params) {
    Grid next = convolve(grid, kernel); // holds U, overwritten in place below
    for (std::size_t i = 0; i < next.cells.size(); ++i) {
        next.cells[i] = std::clamp(grid.cells[i] + params.dt * growth(params, next.cells[i]), 0.0f, 1.0f);
    }
    return next;
}

} // namespace lenia
