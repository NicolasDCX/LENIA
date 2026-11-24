#include "lenia/step.hpp"

#include <cmath>
#include <cstdio>

namespace {

// Steps `grid` with a 1x1 identity kernel, so U equals the current value, and compares.
int check(const char* name, const lenia::Grid& grid, const lenia::GrowthParams& params, const float* expected) {
    const lenia::Grid original = grid;
    lenia::Grid kernel(1, 1, 1.0f);
    const lenia::Grid out = lenia::step_lenia(grid, kernel, params);
    const float tol = 0.00001f;

    std::printf("%s: [", name);
    for (std::size_t i = 0; i < out.cells.size(); ++i) {
        std::printf(i ? ", %.7f" : "%.7f", static_cast<double>(out.cells[i]));
    }
    std::printf("]\n");

    int failures = 0;
    for (std::size_t i = 0; i < grid.cells.size(); ++i) {
        if (std::fabs(out.cells[i] - expected[i]) > tol) {
            std::printf("%s: MISMATCH at %zu: got %.7f, expected %.7f\n", name, i,
                        static_cast<double>(out.cells[i]), static_cast<double>(expected[i]));
            ++failures;
        }
        if (grid.cells[i] != original.cells[i]) {
            std::printf("%s: INPUT MODIFIED at %zu\n", name, i);
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;

    if (lenia::GrowthParams{}.function != lenia::GrowthFunction::Gaussian) {
        std::printf("MISMATCH: default growth function is not Gaussian\n");
        ++failures;
    }

    // 4x1 row: clamp at zero, peak growth, growth away from the peak, 3 sigma from the peak.
    lenia::Grid grid(4, 1, 0.0f);
    grid.cells = {0.0f, 0.5f, 0.6f, 0.8f};

    // Gaussian: 0.6 -> 0.6 + 0.1 * (2 exp(-0.5) - 1); 0.8 -> 0.8 + 0.1 * (2 exp(-4.5) - 1).
    const lenia::GrowthParams gaussian{0.5f, 0.1f, 0.1f}; // mu, sigma, dt; Gaussian by default
    const float gaussian_expected[4] = {0.0f, 0.6f, 0.6213061f, 0.7022218f};
    failures += check("gaussian", grid, gaussian, gaussian_expected);

    // Polynomial: 0.6 -> 0.6 + 0.1 * (2 (1 - 1/9)^4 - 1); at 3 sigma growth bottoms out at -1.
    const lenia::GrowthParams polynomial{0.5f, 0.1f, 0.1f, lenia::GrowthFunction::Polynomial};
    const float polynomial_expected[4] = {0.0f, 0.6f, 0.6248590f, 0.7f};
    failures += check("polynomial", grid, polynomial, polynomial_expected);

    // growth() is what step_lenia applies.
    if (std::fabs(lenia::growth(polynomial, 0.5f) - 1.0f) > 1e-6f ||
        std::fabs(lenia::growth(polynomial, 0.9f) + 1.0f) > 1e-6f ||
        std::fabs(lenia::growth(gaussian, 0.5f) - 1.0f) > 1e-6f) {
        std::printf("MISMATCH: growth() peak or floor\n");
        ++failures;
    }

    std::printf(failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
