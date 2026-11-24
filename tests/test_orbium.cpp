// The bundled Orbium (patterns/orbium.lenia, upstream O2u) must load with its upstream rule
// (polynomial core and polynomial growth) and, run on the CPU, survive and glide.

#include "lenia/kernel_shape.hpp"
#include "lenia/pattern.hpp"
#include "lenia/step.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAILED: %s\n", what);
        ++failures;
    }
}

double mass(const lenia::Grid& g) {
    double m = 0.0;
    for (float c : g.cells) m += c;
    return m;
}

// Center of mass on the torus: the circular mean of each coordinate, in cells.
struct Point {
    double x, y;
};
Point center_of_mass(const lenia::Grid& g) {
    const double two_pi = 6.283185307179586;
    double cx = 0, sx = 0, cy = 0, sy = 0;
    for (std::size_t y = 0; y < g.height; ++y) {
        for (std::size_t x = 0; x < g.width; ++x) {
            const double m = g.at(x, y);
            const double ax = two_pi * static_cast<double>(x) / static_cast<double>(g.width);
            const double ay = two_pi * static_cast<double>(y) / static_cast<double>(g.height);
            cx += m * std::cos(ax);
            sx += m * std::sin(ax);
            cy += m * std::cos(ay);
            sy += m * std::sin(ay);
        }
    }
    const auto unwrap = [&](double c, double s, std::size_t n) {
        double a = std::atan2(s, c);
        if (a < 0) a += two_pi;
        return a / two_pi * static_cast<double>(n);
    };
    return {unwrap(cx, sx, g.width), unwrap(cy, sy, g.height)};
}

// Shortest signed difference b - a on a ring of n cells.
double ring_delta(double a, double b, std::size_t n) {
    const double size = static_cast<double>(n);
    double d = std::fmod(b - a, size);
    if (d > size / 2) d -= size;
    if (d < -size / 2) d += size;
    return d;
}

} // namespace

int main() {
    lenia::Pattern p;
    try {
        p = lenia::load_pattern_file(std::string(LENIA_PATTERN_DIR) + "/orbium.lenia");
    } catch (const std::exception& e) {
        std::printf("FAILED: %s\n", e.what());
        return 1;
    }

    // Upstream O2u: R = 13, T = 10, m = 0.15, s = 0.015, b = [1], kn = 1, gn = 1.
    expect(p.kernel.radius == 13, "R = 13");
    expect(p.kernel.rings.size() == 1 && p.kernel.rings[0] == 1.0f, "rings = [1]");
    expect(p.kernel.core == lenia::KernelCore::Polynomial, "polynomial kernel core");
    expect(p.growth.function == lenia::GrowthFunction::Polynomial, "polynomial growth");
    expect(std::fabs(p.growth.mu - 0.15f) < 1e-6f && std::fabs(p.growth.sigma - 0.015f) < 1e-6f, "mu, sigma");
    expect(std::fabs(p.growth.dt - 0.1f) < 1e-6f, "T = 10");
    expect(!p.source.empty() && !p.license.empty(), "source and license recorded");
    expect(p.cells.width >= 15 && p.cells.width <= 24 && p.cells.height >= 15 && p.cells.height <= 24,
           "cells are roughly 20x20");
    if (failures != 0) {
        std::printf("FAIL\n");
        return 1;
    }

    // 64x64 leaves the creature plus the kernel's reach (2R + ~20 cells) clear of its own
    // wrapped image.
    const lenia::Grid kernel = lenia::normalize_kernel(lenia::make_kernel(p.kernel), lenia::KernelNormalization::Sum);
    lenia::Grid world(64, 64, 0.0f);
    lenia::stamp_pattern(world, p.cells, world.width / 2, world.height / 2);
    const double initial_mass = mass(world);

    // Warm up (the stamped cells settle into the steady glider), then track the center of
    // mass step by step so wrapping around the torus does not hide the motion.
    constexpr int kWarmup = 50;
    constexpr int kTracked = 200;
    for (int g = 0; g < kWarmup; ++g) world = lenia::step_lenia(world, kernel, p.growth);
    const double settled_mass = mass(world);

    double min_mass = settled_mass;
    double max_mass = settled_mass;
    Point prev = center_of_mass(world);
    double dx = 0.0, dy = 0.0;
    for (int g = 0; g < kTracked; ++g) {
        world = lenia::step_lenia(world, kernel, p.growth);
        const double m = mass(world);
        min_mass = std::fmin(min_mass, m);
        max_mass = std::fmax(max_mass, m);
        const Point c = center_of_mass(world);
        dx += ring_delta(prev.x, c.x, world.width);
        dy += ring_delta(prev.y, c.y, world.height);
        prev = c;
    }

    std::size_t occupied = 0;
    for (float c : world.cells) occupied += c > 0.05f ? 1 : 0;
    const double distance = std::hypot(dx, dy);

    std::printf("mass: initial %.2f, after warmup %.2f, range over %d generations [%.2f, %.2f]\n", initial_mass,
                settled_mass, kTracked, min_mass, max_mass);
    std::printf("center of mass moved (%.2f, %.2f) = %.2f cells, %.4f cells/generation\n", dx, dy, distance,
                distance / kTracked);
    std::printf("occupied cells (> 0.05) at the end: %zu of %zu\n", occupied, world.cells.size());

    // Survival: the mass neither dies out nor explodes, and the creature stays compact.
    expect(settled_mass > 0.5 * initial_mass && settled_mass < 2.0 * initial_mass, "mass survives warmup");
    expect(min_mass > 0.7 * settled_mass && max_mass < 1.3 * settled_mass, "mass stays steady while gliding");
    expect(occupied > 0 && occupied < world.cells.size() / 8, "creature stays compact");
    // Motion: a dead, stationary or oscillating blob moves well under a cell in 200 generations.
    expect(distance > 3.0, "center of mass glides at least 3 cells");

    std::printf(failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
