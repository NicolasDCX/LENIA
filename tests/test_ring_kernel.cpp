#include "lenia/convolve.hpp"
#include "lenia/kernel.hpp"

#include <cmath>
#include <cstdio>

int main() {
    int failures = 0;
    const float tol = 0.00001f;

    const lenia::Grid k = lenia::make_ring_kernel(2.0f, 1.0f);

    std::printf("Kernel (%zux%zu):\n", k.width, k.height);
    for (std::size_t y = 0; y < k.height; ++y) {
        for (std::size_t x = 0; x < k.width; ++x) {
            std::printf("%c ", k.at(x, y) > 0.0f ? '#' : '.');
        }
        std::printf("\n");
    }

    std::size_t positive = 0;
    float sum = 0.0f;
    for (float w : k.cells) {
        sum += w;
        if (w > 0.0f) {
            ++positive;
            if (std::fabs(w - 1.0f / 12.0f) > tol) {
                std::printf("WRONG WEIGHT: %.7f\n", static_cast<double>(w));
                ++failures;
            }
        }
    }
    const float center = k.at(k.width / 2, k.height / 2);
    std::printf("Positive cells: %zu, center: %g, sum: %.7f\n", positive,
                static_cast<double>(center), static_cast<double>(sum));
    if (positive != 12) { std::printf("EXPECTED 12 POSITIVE CELLS\n"); ++failures; }
    if (center != 0.0f) { std::printf("CENTER NOT ZERO\n"); ++failures; }
    if (std::fabs(sum - 1.0f) > tol) { std::printf("SUM NOT 1\n"); ++failures; }

    // Uniform density in, same density out (including at wrapped edges).
    const lenia::Grid grid(9, 9, 0.4f);
    const lenia::Grid out = lenia::convolve(grid, k);
    float max_err = 0.0f;
    for (float v : out.cells) {
        max_err = std::fmax(max_err, std::fabs(v - 0.4f));
    }
    std::printf("Uniform 0.4 grid: max |out - 0.4| = %g\n", static_cast<double>(max_err));
    if (max_err > tol) { std::printf("UNIFORM DENSITY NOT PRESERVED\n"); ++failures; }

    std::printf(failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
