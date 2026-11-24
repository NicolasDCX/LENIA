#include "lenia/kernel_shape.hpp"
#include "lenia/pattern.hpp"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAILED: %s\n", what);
        ++failures;
    }
}

bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

template <class F>
bool throws_invalid(F f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    using namespace lenia;

    // Cores: zero at the ends, peak 1 in the middle.
    for (KernelCore core : {KernelCore::ExponentialBump, KernelCore::Polynomial, KernelCore::Step,
                            KernelCore::GaussianBell}) {
        expect(kernel_core(core, 0.0f) == 0.0f, "core is 0 at t = 0");
        expect(kernel_core(core, 1.0f) == 0.0f, "core is 0 at t = 1");
        expect(near(kernel_core(core, 0.5f), 1.0), "core peaks at 1 for t = 0.5");
    }

    // Single positive ring: size, center, symmetry, nothing outside r < 1.
    KernelShape one;
    one.radius = 13;
    const Grid k1 = make_kernel(one);
    expect(k1.width == 27 && k1.height == 27, "radius 13 gives a 27x27 kernel");
    expect(k1.at(13, 13) == 0.0f, "center weight is 0");
    expect(k1.at(0, 13) == 0.0f && k1.at(13, 0) == 0.0f, "r = 1 is outside the kernel");
    expect(k1.at(0, 0) == 0.0f, "corner is outside the kernel");
    bool symmetric = true;
    for (std::size_t y = 0; y < k1.height; ++y)
        for (std::size_t x = 0; x < k1.width; ++x)
            symmetric = symmetric && k1.at(x, y) == k1.at(k1.width - 1 - x, y) && k1.at(x, y) == k1.at(y, x);
    expect(symmetric, "single ring kernel is mirror and diagonal symmetric");
    expect(kernel_stats(k1).negative == 0.0, "positive ring has no negative weights");

    const Grid n1 = normalize_kernel(k1, KernelNormalization::Sum);
    expect(near(kernel_stats(n1).sum, 1.0), "Sum normalization sums to 1");

    // Signed two-ring kernel: inner excites, outer inhibits.
    KernelShape two;
    two.radius = 10;
    two.rings = {1.0f, -0.5f};
    expect(kernel_shell(two, 0.25f) > 0.0f, "inner ring is positive");
    expect(kernel_shell(two, 0.75f) < 0.0f, "outer ring is negative");
    expect(near(kernel_shell(two, 0.75f), -0.5), "outer ring peaks at its weight");
    const Grid k2 = make_kernel(two);
    const KernelStats s2 = kernel_stats(k2);
    expect(s2.positive > 0.0 && s2.negative < 0.0, "signed kernel has both signs");

    const KernelStats p2 = kernel_stats(normalize_kernel(k2, KernelNormalization::Positive));
    expect(near(p2.positive, 1.0), "Positive normalization: positive part sums to 1");
    expect(near(p2.negative, s2.negative / s2.positive), "Positive normalization keeps the ratio");

    const KernelStats a2 = kernel_stats(normalize_kernel(k2, KernelNormalization::AbsoluteSum));
    expect(near(a2.positive - a2.negative, 1.0), "AbsoluteSum normalization: |w| sums to 1");

    const Grid none = normalize_kernel(k2, KernelNormalization::None);
    expect(none.cells == k2.cells, "None leaves weights unchanged");

    // Sum normalization must not flip signs or divide by cancellation noise.
    Grid cancel(3, 3, 0.0f);
    cancel.at(0, 1) = 1.0f;
    cancel.at(2, 1) = -1.0f;
    expect(throws_invalid([&] { normalize_kernel(cancel, KernelNormalization::Sum); }),
           "Sum throws when weights cancel");
    Grid negative(3, 3, 0.0f);
    negative.at(1, 0) = -2.0f;
    expect(throws_invalid([&] { normalize_kernel(negative, KernelNormalization::Sum); }),
           "Sum throws for a negative sum");
    expect(throws_invalid([&] { normalize_kernel(Grid(3, 3, 0.0f), KernelNormalization::AbsoluteSum); }),
           "AbsoluteSum throws for an all-zero kernel");

    KernelShape bad;
    bad.rings.clear();
    expect(throws_invalid([&] { make_kernel(bad); }), "no rings throws");
    bad.rings = {1.0f};
    bad.radius = 0;
    expect(throws_invalid([&] { make_kernel(bad); }), "radius 0 throws");

    // Pattern parsing: Python-style nested list, T instead of dt, comments.
    std::istringstream text(
        "# test pattern\n"
        "name Test creature\n"
        "source https://example.org/data\n"
        "license CC0\n"
        "R 7\n"
        "T 5\n"
        "mu 0.2\n"
        "sigma 0.02\n"
        "rings 1 -0.25\n"
        "core polynomial\n"
        "growth polynomial\n"
        "cells [[0, 0.5, 1],\n"
        "       [0.25],   # short row\n"
        "       [1, 0, 0, 0.75]]\n");
    const Pattern p = parse_pattern(text);
    expect(p.name == "Test creature", "pattern name");
    expect(p.source == "https://example.org/data", "pattern source");
    expect(p.license == "CC0", "pattern license");
    expect(p.kernel.radius == 7, "pattern R");
    expect(near(p.growth.dt, 0.2), "pattern T converts to dt");
    expect(near(p.growth.mu, 0.2) && near(p.growth.sigma, 0.02), "pattern mu, sigma");
    expect(p.kernel.rings.size() == 2 && p.kernel.rings[1] == -0.25f, "pattern rings");
    expect(p.kernel.core == KernelCore::Polynomial, "pattern core");
    expect(p.growth.function == GrowthFunction::Polynomial, "pattern growth");
    expect(p.cells.width == 4 && p.cells.height == 3, "pattern cells padded to 4x3");
    expect(p.cells.at(2, 0) == 1.0f && p.cells.at(0, 1) == 0.25f && p.cells.at(1, 1) == 0.0f &&
               p.cells.at(3, 2) == 0.75f,
           "pattern cell values");

    std::istringstream plain("cells\n1 2\n3 4\n");
    const Pattern q = parse_pattern(plain);
    expect(q.cells.width == 2 && q.cells.height == 2 && q.cells.at(0, 1) == 3.0f, "bracket-free rows");
    expect(q.growth.function == GrowthFunction::Gaussian, "growth defaults to Gaussian");

    std::istringstream unknown("speed 3\ncells 1\n");
    expect(throws_invalid([&] { parse_pattern(unknown); }), "unknown key throws");
    std::istringstream bad_growth("growth step\ncells 1\n");
    expect(throws_invalid([&] { parse_pattern(bad_growth); }), "unknown growth throws");
    std::istringstream nocells("R 5\n");
    expect(throws_invalid([&] { parse_pattern(nocells); }), "missing cells throws");

    // Stamping wraps around the world edges.
    Grid world(5, 5, 0.0f);
    Grid stamp(3, 3, 0.0f);
    stamp.at(0, 0) = 1.0f;
    stamp.at(2, 2) = 2.0f;
    stamp_pattern(world, stamp, 0, 0);
    expect(world.at(4, 4) == 1.0f && world.at(1, 1) == 2.0f, "stamp centered at origin wraps");

    std::printf(failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
