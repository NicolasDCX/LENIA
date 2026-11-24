#include "lenia/convolve.hpp"

#include <cstdio>

static void print_grid(const char* label, const lenia::Grid& g) {
    std::printf("%s\n", label);
    for (std::size_t y = 0; y < g.height; ++y) {
        for (std::size_t x = 0; x < g.width; ++x) {
            std::printf("%4g", static_cast<double>(g.at(x, y)));
        }
        std::printf("\n");
    }
}

int main() {
    // 5x5 grid of zeros with a single 1 in the top-left corner.
    lenia::Grid grid(5, 5, 0.0f);
    grid.at(0, 0) = 1.0f;

    // 3x3 kernel: 2 at center, 3 right of center, 4 above center.
    lenia::Grid kernel(3, 3, 0.0f);
    kernel.at(1, 1) = 2.0f; // center
    kernel.at(2, 1) = 3.0f; // right
    kernel.at(1, 0) = 4.0f; // above

    const lenia::Grid out = lenia::convolve(grid, kernel);

    // Expected: 2 top-left, 3 to its right, 4 bottom-left (top edge wraps).
    lenia::Grid expected(5, 5, 0.0f);
    expected.at(0, 0) = 2.0f;
    expected.at(1, 0) = 3.0f;
    expected.at(0, 4) = 4.0f;

    print_grid("Result:", out);

    int failures = 0;
    for (std::size_t y = 0; y < 5; ++y) {
        for (std::size_t x = 0; x < 5; ++x) {
            if (out.at(x, y) != expected.at(x, y)) {
                std::printf("MISMATCH at (x=%zu, y=%zu): got %g, expected %g\n", x, y,
                            static_cast<double>(out.at(x, y)),
                            static_cast<double>(expected.at(x, y)));
                ++failures;
            }
        }
    }

    std::printf(failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
