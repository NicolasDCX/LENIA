#include "lenia/kernel.hpp"
#include "lenia/kernel_shape.hpp"
#include "lenia/node_graph.hpp"
#include "lenia/pattern.hpp"
#include "lenia/rule.hpp"
#include "lenia/step.hpp"
#include "lenia/vulkan_step.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

// Return code CTest treats as "skipped" (SKIP_RETURN_CODE in CMakeLists.txt).
constexpr int kSkip = 77;

// GPU may fuse multiply-adds and uses its own exp; see vulkan_step.hpp.
constexpr float kTol = 0.0001f;

// Deterministic values in [0, 1).
lenia::Grid random_grid(std::size_t w, std::size_t h, std::uint32_t seed) {
    lenia::Grid g(w, h);
    for (float& c : g.cells) {
        seed = seed * 1664525u + 1013904223u;
        c = static_cast<float>(seed >> 8) / 16777216.0f;
    }
    return g;
}

// generations == 1 uses the single-step overload; otherwise the batched one, checked against
// the CPU step applied `generations` times.
int compare(const char* name, const lenia::Grid& grid, const lenia::Grid& kernel,
            const lenia::GrowthParams& params, lenia::VulkanStepper& gpu, std::size_t generations = 1) {
    lenia::Grid cpu_out = grid;
    for (std::size_t g = 0; g < generations; ++g) cpu_out = lenia::step_lenia(cpu_out, kernel, params);
    const lenia::Grid gpu_out =
        generations == 1 ? gpu.step_lenia(grid, kernel, params) : gpu.step_lenia(grid, kernel, params, generations);

    if (gpu_out.width != cpu_out.width || gpu_out.height != cpu_out.height ||
        gpu_out.cells.size() != cpu_out.cells.size()) {
        std::printf("%s: SHAPE MISMATCH gpu %zux%zu, cpu %zux%zu\n", name, gpu_out.width, gpu_out.height,
                    cpu_out.width, cpu_out.height);
        return 1;
    }

    int failures = 0;
    float max_diff = 0.0f;
    for (std::size_t y = 0; y < cpu_out.height; ++y) {
        for (std::size_t x = 0; x < cpu_out.width; ++x) {
            const float diff = std::fabs(gpu_out.at(x, y) - cpu_out.at(x, y));
            if (diff > max_diff) max_diff = diff;
            if (!(diff <= kTol)) { // also catches NaN
                if (failures < 10) {
                    std::printf("%s: MISMATCH at (%zu, %zu): gpu %.7f, cpu %.7f\n", name, x, y,
                                static_cast<double>(gpu_out.at(x, y)), static_cast<double>(cpu_out.at(x, y)));
                }
                ++failures;
            }
        }
    }
    std::printf("%s: %zux%zu grid, %zux%zu kernel, %zu generation(s), max |gpu - cpu| = %.3g, %d mismatches\n",
                name, grid.width, grid.height, kernel.width, kernel.height, generations,
                static_cast<double>(max_diff), failures);
    return failures;
}

// Node-graph rule on the GPU (VulkanStepper::step_rule, lenia_rule.glsl) against the CPU
// step_rule, `generations` times with the same program; `tol` 0 demands exact equality.
int compare_rule(const char* name, const lenia::Grid& grid, const lenia::Grid& kernel,
                 const lenia::RuleProgram& program, lenia::VulkanStepper& gpu, std::size_t generations, float tol) {
    lenia::Grid cpu_out = grid;
    for (std::size_t g = 0; g < generations; ++g) cpu_out = lenia::step_rule(cpu_out, kernel, program);
    const lenia::Grid gpu_out = gpu.step_rule(grid, kernel, program, generations);
    if (gpu_out.cells.size() != cpu_out.cells.size()) {
        std::printf("%s: SHAPE MISMATCH\n", name);
        return 1;
    }
    int failures = 0;
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < cpu_out.cells.size(); ++i) {
        const float diff = std::fabs(gpu_out.cells[i] - cpu_out.cells[i]);
        if (diff > max_diff) max_diff = diff;
        if (!(diff <= tol)) {
            if (failures < 10) {
                std::printf("%s: MISMATCH at %zu: gpu %.7f, cpu %.7f\n", name, i, static_cast<double>(gpu_out.cells[i]),
                            static_cast<double>(cpu_out.cells[i]));
            }
            ++failures;
        }
    }
    std::printf("%s: %zux%zu grid, %zu ops, %zu generation(s), max |gpu - cpu| = %.3g, %d mismatches\n", name,
                grid.width, grid.height, program.ops.size(), generations, static_cast<double>(max_diff), failures);
    return failures;
}

lenia::RuleProgram compiled(const lenia::Graph& g, std::uint64_t generation, const lenia::GrowthParams& panel);

// compare() through the FFT path (VulkanStepper::set_convolution), which must actually run
// (`expect_fft`) or fall back to Direct for grids it does not support.
int compare_fft(const char* name, const lenia::Grid& grid, const lenia::Grid& kernel,
                const lenia::GrowthParams& params, lenia::VulkanStepper& gpu, std::size_t generations,
                bool expect_fft = true) {
    gpu.set_convolution(lenia::ConvolutionMethod::Fft);
    int failures = compare(name, grid, kernel, params, gpu, generations);
    const bool used_fft = gpu.last_convolution() == lenia::ConvolutionMethod::Fft;
    gpu.set_convolution(lenia::ConvolutionMethod::Direct);
    if (used_fft != expect_fft) {
        std::printf("%s: MISMATCH, ran %s, expected %s\n", name, used_fft ? "FFT" : "Direct",
                    expect_fft ? "FFT" : "the Direct fallback");
        ++failures;
    }
    return failures;
}

bool same_bits(const lenia::Grid& a, const lenia::Grid& b) {
    return a.cells.size() == b.cells.size() &&
           std::memcmp(a.cells.data(), b.cells.data(), a.cells.size() * sizeof(float)) == 0;
}

// The FFT path (shaders/fft_convolve.comp) against the CPU, and its kernel spectrum cache.
int fft_cases(lenia::VulkanStepper& gpu, const lenia::Grid& ring, const lenia::Grid& asym,
              const lenia::GrowthParams& params) {
    const lenia::VulkanDeviceInfo& dev = gpu.device_info();
    std::printf("fft: longest supported side %zu\n", dev.fft_max_size);
    if (!gpu.fft_supported(64, 32)) {
        // Every conforming device has 16 KiB of shared memory, enough for 2048.
        std::printf("fft: MISMATCH, 64x32 unsupported on this device\n");
        return 1;
    }
    int failures = 0;
    lenia::GrowthParams poly = params;
    poly.function = lenia::GrowthFunction::Polynomial;

    // Orientation: an asymmetric kernel stamped from an impulse near a corner (wraps on both
    // axes), with growth steep enough that a flipped stamp would fail.
    lenia::Grid impulse(32, 16, 0.0f);
    impulse.at(1, 14) = 1.0f;
    failures += compare_fft("fft_asym_impulse_32x16", impulse, asym, lenia::GrowthParams{0.15f, 0.1f, 0.5f}, gpu, 1);
    // Rectangular both ways, both growth functions.
    failures += compare_fft("fft_asym_64x16", random_grid(64, 16, 31u), asym, params, gpu, 1);
    failures += compare_fft("fft_asym_16x64_poly", random_grid(16, 64, 32u), asym, poly, gpu, 1);
    // Kernel larger than the grid: wraps more than once on both axes (15x15 on 8x8).
    failures += compare_fft("fft_ring_8x8", random_grid(8, 8, 33u), ring, params, gpu, 1);
    failures += compare_fft("fft_ring_64x8", random_grid(64, 8, 34u), ring, params, gpu, 1);
    // Repeated steps in one batch: even and odd counts end in different buffers.
    failures += compare_fft("fft_ring_64x64_gen2", random_grid(64, 64, 35u), ring, params, gpu, 2);
    failures += compare_fft("fft_ring_64x64_gen5", random_grid(64, 64, 36u), ring, params, gpu, 5);
    failures += compare_fft("fft_asym_32x32_gen4_poly", random_grid(32, 32, 37u), asym, poly, gpu, 4);
    // Unsupported grids fall back to Direct and still match.
    failures += compare_fft("fft_fallback_37x23", random_grid(37, 23, 38u), ring, params, gpu, 1, false);
    failures += compare_fft("fft_fallback_4x8", random_grid(4, 8, 39u), asym, params, gpu, 1, false);

    // The bundled Orbium with its own rule, as in the Direct cases: R13 and R32 (65x65 on a
    // 64x64 grid, so it wraps).
    try {
        const lenia::Pattern orbium = lenia::load_pattern_file(std::string(LENIA_PATTERN_DIR) + "/orbium.lenia");
        const lenia::Grid k13 =
            lenia::normalize_kernel(lenia::make_kernel(orbium.kernel), lenia::KernelNormalization::Sum);
        lenia::KernelShape shape = orbium.kernel;
        shape.radius = 32;
        const lenia::Grid k32 = lenia::normalize_kernel(lenia::make_kernel(shape), lenia::KernelNormalization::Sum);
        lenia::Grid world(64, 64, 0.0f);
        lenia::stamp_pattern(world, orbium.cells, 32, 32);
        failures += compare_fft("fft_orbium_64x64", world, k13, orbium.growth, gpu, 1);
        for (int g = 0; g < 30; ++g) world = lenia::step_lenia(world, k13, orbium.growth);
        failures += compare_fft("fft_orbium_64x64_settled", world, k13, orbium.growth, gpu, 1);
        failures += compare_fft("fft_orbium_r32_64x64", world, k32, orbium.growth, gpu, 1);
    } catch (const std::exception& e) {
        std::printf("fft_orbium: FAILED to load: %s\n", e.what());
        ++failures;
    }

    // Kernel spectrum cache: alternating kernels, and one kernel at two grid sizes, must each
    // match the CPU (a stale spectrum would not); an unchanged kernel reuses the spectrum.
    gpu.set_convolution(lenia::ConvolutionMethod::Fft);
    {
        const lenia::Grid g = random_grid(32, 32, 40u);
        const bool timed = dev.timestamps;
        int cache_failures = 0;
        cache_failures += compare("fft_cache_ring", g, ring, params, gpu);
        const bool first_transformed = gpu.last_kernel_transform_ms().has_value();
        cache_failures += compare("fft_cache_ring_again", g, ring, params, gpu);
        const bool second_transformed = gpu.last_kernel_transform_ms().has_value();
        cache_failures += compare("fft_cache_asym", g, asym, params, gpu);
        const bool third_transformed = gpu.last_kernel_transform_ms().has_value();
        cache_failures += compare("fft_cache_ring_after_asym", g, ring, params, gpu);
        cache_failures += compare("fft_cache_ring_64x16", random_grid(64, 16, 41u), ring, params, gpu);
        cache_failures += compare("fft_cache_ring_32x32_again", g, ring, params, gpu);
        // Same size, one weight changed: a new kernel, not a cache hit.
        lenia::Grid ring2 = ring;
        ring2.at(7, 1) += 0.05f;
        cache_failures += compare("fft_cache_ring_edited", g, ring2, params, gpu);
        const bool timing_ok = !timed || (first_transformed && !second_transformed && third_transformed);
        std::printf("fft_cache: transform on first use %s, reused when unchanged %s, redone for a new kernel %s\n",
                    timed ? (first_transformed ? "yes" : "NO") : "untimed",
                    timed ? (!second_transformed ? "yes" : "NO") : "untimed",
                    timed ? (third_transformed ? "yes" : "NO") : "untimed");
        failures += cache_failures + (timing_ok ? 0 : 1);

        // A batch of 2 equals two single FFT steps bit for bit (same dispatches, cached or not).
        const lenia::Grid one = gpu.step_lenia(g, ring, params);
        const bool exact = same_bits(gpu.step_lenia(g, ring, params, 2), gpu.step_lenia(one, ring, params));
        std::printf("fft_batch_of_2: %s\n", exact ? "== 2 single steps" : "MISMATCH");
        failures += exact ? 0 : 1;

        // FFT against Direct on the GPU at a larger size, where the CPU would be slow to check.
        const lenia::Grid big = random_grid(256, 256, 42u);
        const lenia::Grid via_fft = gpu.step_lenia(big, ring, params, 3);
        gpu.set_convolution(lenia::ConvolutionMethod::Direct);
        const lenia::Grid via_direct = gpu.step_lenia(big, ring, params, 3);
        float max_diff = 0.0f;
        for (std::size_t i = 0; i < via_fft.cells.size(); ++i) {
            max_diff = std::max(max_diff, std::fabs(via_fft.cells[i] - via_direct.cells[i]));
        }
        const bool close = !(max_diff > kTol) && via_fft.cells.size() == via_direct.cells.size();
        std::printf("fft_vs_direct_256x256_gen3: max |fft - direct| = %.3g, %s\n", static_cast<double>(max_diff),
                    close ? "ok" : "MISMATCH");
        failures += close ? 0 : 1;
    }
    gpu.set_convolution(lenia::ConvolutionMethod::Direct);

    // step_rule ignores the setting.
    gpu.set_convolution(lenia::ConvolutionMethod::Fft);
    gpu.step_rule(random_grid(32, 32, 43u), lenia::moore_kernel(),
                  compiled(lenia::make_preset(lenia::GraphPreset::GameOfLife), 0, {}), 1);
    const bool rule_direct = gpu.last_convolution() == lenia::ConvolutionMethod::Direct;
    gpu.set_convolution(lenia::ConvolutionMethod::Direct);
    std::printf("fft_rule_uses_direct: %s\n", rule_direct ? "ok" : "MISMATCH");
    failures += rule_direct ? 0 : 1;
    return failures;
}

lenia::RuleProgram compiled(const lenia::Graph& g, std::uint64_t generation, const lenia::GrowthParams& panel) {
    const lenia::RuleCompile c = lenia::compile_rule(g, generation, panel);
    if (!c.linked || !c.error.empty()) throw std::runtime_error("rule does not compile: " + c.error);
    return c.program;
}

// A rule using every op: next = Select(U >= mu, Clamp(A + dt * (G_own(U) + field), 0, 1),
// A * 0.95) with the Life node's opinion added at weight 0.01, the environment at strength 0.05.
lenia::Graph rewired_rule() {
    using namespace lenia;
    Graph g;
    const int density = g.add_node(NodeKind::Density);
    const int nb = g.add_node(NodeKind::Neighborhood);
    const int growth = g.add_node(NodeKind::Growth);
    g.set_param(growth, growth_node::Source, static_cast<float>(growth_node::Own));
    g.set_param(growth, growth_node::Function, 1.0f);
    g.set_param(growth, growth_node::Mu, 0.45f);
    g.set_param(growth, growth_node::Sigma, 0.08f);
    g.set_param(growth, growth_node::Dt, 0.2f);
    g.connect(nb, 0, growth, growth_node::UIn);
    const int env = g.add_node(NodeKind::Environment);
    g.set_param(env, environment_node::Pattern, static_cast<float>(EnvironmentPattern::Ripples));
    g.set_param(env, environment_node::Strength, 0.05f);
    g.set_param(env, environment_node::Scale, 9.0f);
    g.set_param(env, environment_node::Angle, 30.0f);
    const int rate = g.add_node(NodeKind::Add);
    g.connect(growth, growth_node::GOut, rate, add_node::A);
    g.connect(env, environment_node::FieldOut, rate, add_node::B);
    const int scaled = g.add_node(NodeKind::Multiply);
    g.connect(rate, 0, scaled, multiply_node::A);
    g.connect(growth, growth_node::DtOut, scaled, multiply_node::B); // Value -> Cell
    const int sum = g.add_node(NodeKind::Add);
    g.connect(density, 0, sum, add_node::A);
    g.connect(scaled, 0, sum, add_node::B);
    const int life = g.add_node(NodeKind::Life); // reads the density and U unlinked
    const int nudge = g.add_node(NodeKind::Multiply);
    g.set_param(nudge, multiply_node::B, 0.01f);
    g.connect(life, 0, nudge, multiply_node::A);
    const int nudged = g.add_node(NodeKind::Add);
    g.connect(sum, 0, nudged, add_node::A);
    g.connect(nudge, 0, nudged, add_node::B);
    const int clamp = g.add_node(NodeKind::Clamp);
    g.connect(nudged, 0, clamp, clamp_node::X);
    const int decay = g.add_node(NodeKind::Multiply);
    g.set_param(decay, multiply_node::B, 0.95f);
    g.connect(density, 0, decay, multiply_node::A);
    const int test = g.add_node(NodeKind::Compare);
    g.set_param(test, compare_node::Op, static_cast<float>(CompareOp::GreaterEqual));
    g.set_param(test, compare_node::B, 0.45f);
    g.connect(nb, 0, test, compare_node::AIn);
    const int select = g.add_node(NodeKind::Select);
    g.connect(test, 0, select, select_node::Condition);
    g.connect(clamp, 0, select, select_node::IfTrue);
    g.connect(decay, 0, select, select_node::IfFalse);
    const int out = g.add_node(NodeKind::Output);
    g.connect(select, 0, out, output_node::NextIn);
    return g;
}

} // namespace

int main() {
    std::optional<lenia::VulkanStepper> stepper;
    try {
        stepper.emplace();
    } catch (const std::runtime_error& e) {
        std::printf("SKIP: %s\n", e.what());
        return kSkip;
    }
    lenia::VulkanStepper& gpu = *stepper;
    std::printf("Device: %s\n", gpu.device_name().c_str());

    // Mu near the mean neighborhood value of a uniform random grid, so growth spans its full range.
    const lenia::GrowthParams params{0.5f, 0.05f, 0.1f}; // mu, sigma, dt
    const lenia::Grid ring = lenia::make_ring_kernel(6.0f, 3.0f); // 15x15

    // Asymmetric 5x3 kernel with zeros: catches flipped or transposed orientation.
    lenia::Grid asym(5, 3, 0.0f);
    asym.cells = {0.10f, 0.00f, 0.05f, 0.20f, 0.00f,
                  0.00f, 0.15f, 0.00f, 0.00f, 0.25f,
                  0.30f, 0.00f, 0.00f, 0.10f, 0.00f};

    lenia::Grid impulse(29, 17, 0.0f);
    impulse.at(1, 15) = 1.0f; // near a corner, so the stamp wraps on both axes

    int failures = 0;
    // Width and height not multiples of the 16x16 workgroup.
    failures += compare("ring_37x23", random_grid(37, 23, 1u), ring, params, gpu);
    // Kernel taller than the grid: wraps more than once vertically.
    failures += compare("ring_41x9", random_grid(41, 9, 2u), ring, params, gpu);
    // Tall rather than wide, reusing (shrinking) the buffers from the previous cases.
    failures += compare("asym_13x50", random_grid(13, 50, 3u), asym, params, gpu);
    failures += compare("asym_impulse_29x17", impulse, asym, lenia::GrowthParams{0.15f, 0.1f, 0.5f}, gpu);

    // Batched generations: an even count reads back from the upload buffer, an odd one from the
    // other, and each needs the barrier between dispatches to see the previous generation.
    failures += compare("ring_37x23_gen2", random_grid(37, 23, 4u), ring, params, gpu, 2);
    failures += compare("ring_37x23_gen3", random_grid(37, 23, 5u), ring, params, gpu, 3);
    failures += compare("asym_13x50_gen2", random_grid(13, 50, 6u), asym, params, gpu, 2);
    failures += compare("asym_13x50_gen3", random_grid(13, 50, 7u), asym, params, gpu, 3);

    // GPU timestamps (lenia_bench): a batch reports a positive device time, a zero-generation
    // call none.
    {
        const lenia::Grid g = random_grid(37, 23, 11u);
        gpu.step_lenia(g, ring, params, 4);
        const std::optional<double> ms = gpu.last_gpu_ms();
        gpu.step_lenia(g, ring, params, 0);
        const bool none_after_zero = !gpu.last_gpu_ms().has_value();
        const bool ok = gpu.device_info().timestamps ? (ms && *ms > 0.0 && none_after_zero) : !ms;
        std::printf("gpu_timestamps: %s, batch of 4 took %.4f ms, %s\n",
                    gpu.device_info().timestamps ? "supported" : "unsupported", ms ? *ms : 0.0,
                    ok ? "ok" : "MISMATCH");
        failures += ok ? 0 : 1;
    }

    // Polynomial growth (push constant selects it), single and batched.
    lenia::GrowthParams poly = params;
    poly.function = lenia::GrowthFunction::Polynomial;
    failures += compare("poly_ring_37x23", random_grid(37, 23, 8u), ring, poly, gpu);
    failures += compare("poly_asym_13x50", random_grid(13, 50, 9u), asym, poly, gpu);
    failures += compare("poly_ring_37x23_gen3", random_grid(37, 23, 10u), ring, poly, gpu, 3);
    // Same grid and kernel with Gaussian growth after a polynomial run: the selector is not sticky.
    failures += compare("gauss_after_poly_37x23", random_grid(37, 23, 8u), ring, params, gpu);

    // The bundled Orbium with its own rule (polynomial core and growth, sigma = 0.015, where
    // growth is steep): one generation from the stamped cells and one from a settled state.
    // Single generations only, since rounding differences compound quickly at this sigma.
    try {
        const lenia::Pattern orbium = lenia::load_pattern_file(std::string(LENIA_PATTERN_DIR) + "/orbium.lenia");
        const lenia::Grid kernel =
            lenia::normalize_kernel(lenia::make_kernel(orbium.kernel), lenia::KernelNormalization::Sum);
        lenia::Grid world(64, 64, 0.0f);
        lenia::stamp_pattern(world, orbium.cells, 32, 32);
        failures += compare("orbium_64x64", world, kernel, orbium.growth, gpu);
        for (int g = 0; g < 30; ++g) world = lenia::step_lenia(world, kernel, orbium.growth);
        failures += compare("orbium_64x64_settled", world, kernel, orbium.growth, gpu);
    } catch (const std::exception& e) {
        std::printf("orbium: FAILED to load: %s\n", e.what());
        ++failures;
    }

    // Node-graph rules through the shared GLSL interpreter.
    try {
        // Rewired rule using every op, single and batched (the program is fixed, so batching is
        // the CPU rule applied repeatedly).
        const lenia::RuleProgram rewired = compiled(rewired_rule(), 0, {});
        failures += compare_rule("rule_rewired_37x23", random_grid(37, 23, 21u), ring, rewired, gpu, 1, kTol);
        failures += compare_rule("rule_rewired_37x23_gen3", random_grid(37, 23, 22u), ring, rewired, gpu, 3, kTol);
        failures += compare_rule("rule_rewired_13x50", random_grid(13, 50, 23u), asym, rewired, gpu, 1, kTol);

        // Game of Life on the raw Moore kernel: whole-number counts, so GPU and CPU agree exactly,
        // over many generations, for both the Life node and the compare/select wiring.
        lenia::Grid soup = random_grid(53, 41, 24u);
        for (float& c : soup.cells) c = c < 0.35f ? 1.0f : 0.0f;
        const lenia::Grid moore = lenia::moore_kernel();
        for (lenia::GraphPreset preset : {lenia::GraphPreset::GameOfLife, lenia::GraphPreset::GameOfLifeCompare}) {
            const lenia::RuleProgram life = compiled(lenia::make_preset(preset), 0, {});
            const char* name = preset == lenia::GraphPreset::GameOfLife ? "life_node_53x41_gen40" : "life_compare_53x41_gen40";
            failures += compare_rule(name, soup, moore, life, gpu, 40, 0.0f);
        }
        lenia::Grid glider(16, 16, 0.0f);
        lenia::stamp_pattern(glider, lenia::glider_pattern().cells, 5, 5);
        const lenia::RuleProgram life = compiled(lenia::make_preset(lenia::GraphPreset::GameOfLife), 0, {});
        const lenia::Grid moved = gpu.step_rule(glider, moore, life, 4);
        bool shifted = true;
        for (std::size_t y = 0; y < 16; ++y) {
            for (std::size_t x = 0; x < 16; ++x) shifted = shifted && moved.at((x + 1) % 16, (y + 1) % 16) == glider.at(x, y);
        }
        std::printf("life_glider_gpu_gen4: %s\n", shifted ? "moved (1, 1)" : "MISMATCH");
        failures += shifted ? 0 : 1;

        // The default wired Lenia rule on Orbium, one generation, as for step_lenia above.
        const lenia::Pattern orbium = lenia::load_pattern_file(std::string(LENIA_PATTERN_DIR) + "/orbium.lenia");
        const lenia::Grid kernel =
            lenia::normalize_kernel(lenia::make_kernel(orbium.kernel), lenia::KernelNormalization::Sum);
        lenia::Grid world(64, 64, 0.0f);
        lenia::stamp_pattern(world, orbium.cells, 32, 32);
        for (int g = 0; g < 30; ++g) world = lenia::step_lenia(world, kernel, orbium.growth);
        const lenia::RuleProgram lenia_rule = compiled(lenia::make_preset(lenia::GraphPreset::Idle), 0, orbium.growth);
        failures += compare_rule("rule_lenia_orbium_64x64", world, kernel, lenia_rule, gpu, 1, kTol);
    } catch (const std::exception& e) {
        std::printf("rules: FAILED: %s\n", e.what());
        ++failures;
    }

    // The optional FFT convolution path.
    try {
        failures += fft_cases(gpu, ring, asym, params);
    } catch (const std::exception& e) {
        std::printf("fft: FAILED: %s\n", e.what());
        ++failures;
    }

    std::printf(failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
