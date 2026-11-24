// lenia::convolve_fft, the CPU mirror of the GPU's FFT path (shaders/fft_convolve.comp),
// against the direct lenia::convolve: orientation (asymmetric kernels, impulses), wrapping
// (kernels larger than the grid), rectangular grids, repeated Lenia steps, and the size rules
// shared with the GPU.

#include "lenia/fft_convolve.hpp"
#include "lenia/kernel.hpp"
#include "lenia/kernel_shape.hpp"
#include "lenia/pattern.hpp"
#include "lenia/step.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

// Deterministic values in [0, 1).
lenia::Grid random_grid(std::size_t w, std::size_t h, std::uint32_t seed) {
    lenia::Grid g(w, h);
    for (float& c : g.cells) {
        seed = seed * 1664525u + 1013904223u;
        c = static_cast<float>(seed >> 8) / 16777216.0f;
    }
    return g;
}

// Float rounding of a transform grows with the magnitude of what is summed; 1e-5 of the
// largest possible |U| is far below any orientation or wrapping mistake.
float tolerance(const lenia::Grid& grid, const lenia::Grid& kernel) {
    float abs_sum = 0.0f, max_cell = 0.0f;
    for (float k : kernel.cells) abs_sum += std::fabs(k);
    for (float c : grid.cells) max_cell = std::max(max_cell, std::fabs(c));
    return 1e-5f * std::max(1.0f, abs_sum * max_cell);
}

int compare(const char* name, const lenia::Grid& grid, const lenia::Grid& kernel) {
    const lenia::Grid direct = lenia::convolve(grid, kernel);
    const lenia::Grid fft = lenia::convolve_fft(grid, kernel);
    if (fft.width != direct.width || fft.height != direct.height || fft.cells.size() != direct.cells.size()) {
        std::printf("%s: SHAPE MISMATCH\n", name);
        return 1;
    }
    const float tol = tolerance(grid, kernel);
    int failures = 0;
    float max_diff = 0.0f;
    for (std::size_t y = 0; y < direct.height; ++y) {
        for (std::size_t x = 0; x < direct.width; ++x) {
            const float diff = std::fabs(fft.at(x, y) - direct.at(x, y));
            max_diff = std::max(max_diff, diff);
            if (!(diff <= tol)) { // also catches NaN
                if (failures < 10) {
                    std::printf("%s: MISMATCH at (%zu, %zu): fft %.7f, direct %.7f\n", name, x, y,
                                static_cast<double>(fft.at(x, y)), static_cast<double>(direct.at(x, y)));
                }
                ++failures;
            }
        }
    }
    std::printf("%s: %zux%zu grid, %zux%zu kernel, max |fft - direct| = %.3g (tolerance %.1g), %d mismatches\n", name,
                grid.width, grid.height, kernel.width, kernel.height, static_cast<double>(max_diff),
                static_cast<double>(tol), failures);
    return failures;
}

// `generations` Lenia steps with U from each convolution; rounding compounds, so the growth is
// kept gentle (sigma 0.05) and the tolerance is the GPU parity test's.
int compare_steps(const char* name, lenia::Grid grid, const lenia::Grid& kernel, const lenia::GrowthParams& params,
                  int generations) {
    lenia::Grid a = grid, b = grid;
    for (int g = 0; g < generations; ++g) {
        a = lenia::step_lenia(a, kernel, params);
        const lenia::Grid u = lenia::convolve_fft(b, kernel);
        for (std::size_t i = 0; i < b.cells.size(); ++i) {
            b.cells[i] = std::clamp(b.cells[i] + params.dt * lenia::growth(params, u.cells[i]), 0.0f, 1.0f);
        }
    }
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < a.cells.size(); ++i) max_diff = std::max(max_diff, std::fabs(a.cells[i] - b.cells[i]));
    const bool ok = max_diff <= 1e-4f;
    std::printf("%s: %d generations, max |fft - direct| = %.3g, %s\n", name, generations, static_cast<double>(max_diff),
                ok ? "ok" : "MISMATCH");
    return ok ? 0 : 1;
}

int expect_throw(const char* name, const lenia::Grid& grid, const lenia::Grid& kernel) {
    try {
        lenia::convolve_fft(grid, kernel);
    } catch (const std::invalid_argument&) {
        std::printf("%s: rejected\n", name);
        return 0;
    }
    std::printf("%s: MISMATCH, accepted\n", name);
    return 1;
}

int check(const char* what, bool ok) {
    if (!ok) std::printf("%s: MISMATCH\n", what);
    return ok ? 0 : 1;
}

} // namespace

int main() {
    int failures = 0;

    // Asymmetric 5x3 kernel with zeros and a negative weight: catches flipped or transposed
    // orientation and sign mistakes.
    lenia::Grid asym(5, 3, 0.0f);
    asym.cells = {0.10f, 0.00f, 0.05f, 0.20f, 0.00f,
                  0.00f, 0.15f, 0.00f, -0.05f, 0.25f,
                  0.30f, 0.00f, 0.00f, 0.10f, 0.00f};

    // Impulse near a corner: the stamp is the kernel itself, wrapped on both axes.
    lenia::Grid impulse(16, 8, 0.0f);
    impulse.at(1, 6) = 1.0f;
    failures += compare("asym_impulse_16x8", impulse, asym);
    // Explicitly: a 1 at (1, 6) puts kernel(kx, ky) at (1 + kx - 2, 6 + ky - 1) mod (16, 8).
    {
        const lenia::Grid u = lenia::convolve_fft(impulse, asym);
        int bad = 0;
        for (std::size_t ky = 0; ky < 3; ++ky) {
            for (std::size_t kx = 0; kx < 5; ++kx) {
                const std::size_t x = (1 + 16 + kx - 2) % 16, y = (6 + ky - 1) % 8;
                if (std::fabs(u.at(x, y) - asym.at(kx, ky)) > 1e-6f) ++bad;
            }
        }
        std::printf("asym_impulse_stamp: %s\n", bad == 0 ? "kernel stamped upright at the impulse" : "MISMATCH");
        failures += bad == 0 ? 0 : 1;
    }

    // Rectangular grids, both ways round, and a square one.
    failures += compare("asym_32x16", random_grid(32, 16, 1u), asym);
    failures += compare("asym_16x32", random_grid(16, 32, 2u), asym);
    failures += compare("asym_64x64", random_grid(64, 64, 3u), asym);

    // Kernels larger than the grid: taps wrap around more than once.
    const lenia::Grid ring = lenia::make_ring_kernel(6.0f, 3.0f); // 15x15
    failures += compare("ring_8x8", random_grid(8, 8, 4u), ring);
    failures += compare("ring_32x8", random_grid(32, 8, 5u), ring);
    lenia::Grid tall(3, 21, 0.0f); // taller than a 16-row grid, asymmetric
    for (std::size_t y = 0; y < 21; ++y) tall.at(y % 3, y) = 0.01f * static_cast<float>(y + 1);
    failures += compare("tall_kernel_16x16", random_grid(16, 16, 6u), tall);

    // Orbium's kernel at its own radius and at radius 32 (65x65, larger than 64x64).
    try {
        const lenia::Pattern orbium = lenia::load_pattern_file(std::string(LENIA_PATTERN_DIR) + "/orbium.lenia");
        const lenia::Grid k13 =
            lenia::normalize_kernel(lenia::make_kernel(orbium.kernel), lenia::KernelNormalization::Sum);
        lenia::KernelShape shape = orbium.kernel;
        shape.radius = 32;
        const lenia::Grid k32 = lenia::normalize_kernel(lenia::make_kernel(shape), lenia::KernelNormalization::Sum);
        failures += compare("orbium_r13_128x64", random_grid(128, 64, 7u), k13);
        failures += compare("orbium_r32_64x64", random_grid(64, 64, 8u), k32);

        // Repeated steps from the stamped Orbium, with its own (steep) rule for a few
        // generations, and longer with a gentle one.
        lenia::Grid world(64, 64, 0.0f);
        lenia::stamp_pattern(world, orbium.cells, 32, 32);
        failures += compare_steps("orbium_steps_64x64", world, k13, orbium.growth, 3);
        failures += compare_steps("gentle_steps_64x32", random_grid(64, 32, 9u), k13,
                                  lenia::GrowthParams{0.5f, 0.05f, 0.1f}, 10);
    } catch (const std::exception& e) {
        std::printf("orbium: FAILED: %s\n", e.what());
        ++failures;
    }

    // Unsupported inputs are rejected, not approximated.
    failures += expect_throw("non_power_of_two_12x16", random_grid(12, 16, 10u), asym);
    failures += expect_throw("too_small_4x8", random_grid(4, 8, 11u), asym);
    failures += expect_throw("even_kernel", random_grid(16, 16, 12u), lenia::Grid(4, 3, 0.1f));

    // Size rules and twiddles shared with the GPU path.
    failures += check("fft_max_size(16 KiB) == 2048", lenia::fft_max_size(16384) == 2048);
    failures += check("fft_max_size(48 KiB) == 4096", lenia::fft_max_size(49152) == 4096);
    failures += check("fft_max_size(1 MiB) == kFftMaxSize", lenia::fft_max_size(1 << 20) == lenia::kFftMaxSize);
    failures += check("fft_max_size(32 B) == 0", lenia::fft_max_size(32) == 0);
    failures += check("fft_size_supported(1024, 512)", lenia::fft_size_supported(1024, 512, 2048));
    failures += check("!fft_size_supported(4096, 4096, 2048)", !lenia::fft_size_supported(4096, 4096, 2048));
    failures += check("!fft_size_supported(1000, 1024)", !lenia::fft_size_supported(1000, 1024));
    failures += check("fft_log2(1024) == 10", lenia::fft_log2(1024) == 10);
    failures += check("fft_spectrum_size(16, 8) == 72", lenia::fft_spectrum_size(16, 8) == 72);
    {
        const std::vector<float> t = lenia::fft_twiddles(8); // exp(-2 pi i t / 8), t = 0..3
        const float h = std::sqrt(0.5f);
        const float expected[8] = {1.0f, 0.0f, h, -h, 0.0f, -1.0f, -h, -h};
        bool ok = t.size() == 8;
        for (std::size_t i = 0; ok && i < 8; ++i) ok = std::fabs(t[i] - expected[i]) < 1e-7f;
        failures += check("fft_twiddles(8)", ok);
    }

    std::printf(failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
