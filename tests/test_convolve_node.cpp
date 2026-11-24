// Reusable kernels: Kernel K and Convolve(Field, Kernel) nodes (node_graph.hpp, rule.hpp).
//
// - an unshifted Kernel K is make_kernel(shape) exactly, and a shift moves where a cell looks
//   (+X right, +Y down); kernels past the 65 x 65 limit are refused;
// - independent kernels on Density and Nutrient: next A = K1 ∗ A and next N = K2 ∗ N match
//   lenia::convolve bit for bit, both computed from the current state;
// - a Convolve with the panel kernel's Kernel K reproduces the Neighborhood-based Lenia rule bit
//   for bit; an unlinked Field is the density; the single-field step reads a nutrient of 0;
// - equal (field, kernel) pairs share one convolution, a Kernel K may feed several Convolve
//   nodes, and the limits (missing kernel, more than 4 convolutions) are reported as errors;
// - pack_rule shifts Convolve kernel offsets by the kernel base, pack_rule_kernels packs them;
// - the ShortLongRange demo compiles to two convolutions and turns noise into a pattern;
// - experiment files store both kinds (version 5) and older versions refuse them.

#include "lenia/kernel_shape.hpp"
#include "lenia/node_graph.hpp"
#include "lenia/project.hpp"
#include "lenia/rule.hpp"
#include "lenia/world.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAILED: %s\n", what);
        ++failures;
    }
}

bool same(const lenia::Grid& a, const lenia::Grid& b) {
    if (a.width != b.width || a.height != b.height || a.cells.size() != b.cells.size()) return false;
    for (std::size_t i = 0; i < a.cells.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(a.cells[i]) != std::bit_cast<std::uint32_t>(b.cells[i])) return false;
    }
    return true;
}

// Deterministic values in [0, 1).
lenia::Grid random_grid(std::size_t w, std::size_t h, std::uint32_t seed) {
    lenia::Grid g(w, h);
    for (float& c : g.cells) {
        seed = seed * 1664525u + 1013904223u;
        c = static_cast<float>(seed >> 8) / 16777216.0f;
    }
    return g;
}

// A Kernel K node with the given parameters.
int add_kernel(lenia::Graph& g, int radius, lenia::KernelCore core, std::vector<float> rings,
               lenia::KernelNormalization norm, float shift_x = 0.0f, float shift_y = 0.0f) {
    using namespace lenia::conv_kernel_node;
    const int k = g.add_node(lenia::NodeKind::ConvKernel);
    g.set_param(k, Radius, static_cast<float>(radius));
    g.set_param(k, Core, static_cast<float>(core));
    g.set_param(k, Rings, static_cast<float>(rings.size()));
    for (std::size_t i = 0; i < rings.size(); ++i) g.set_param(k, Ring1 + static_cast<int>(i), rings[i]);
    g.set_param(k, Normalization, static_cast<float>(norm));
    g.set_param(k, ShiftX, shift_x);
    g.set_param(k, ShiftY, shift_y);
    return k;
}

// A Convolve node on `field` (a Density or Nutrient node; -1 leaves Field unlinked) with
// `kernel` (-1 leaves Kernel unlinked).
int add_convolve(lenia::Graph& g, int field, int kernel) {
    const int c = g.add_node(lenia::NodeKind::Convolve);
    if (field >= 0) g.connect(field, lenia::field_node::FieldOut, c, lenia::convolve_node::FieldIn);
    if (kernel >= 0) g.connect(kernel, 0, c, lenia::convolve_node::KernelIn);
    return c;
}

// Centroid of a nonnegative grid.
void centroid(const lenia::Grid& g, double& cx, double& cy) {
    double sum = 0.0, sx = 0.0, sy = 0.0;
    for (std::size_t y = 0; y < g.height; ++y) {
        for (std::size_t x = 0; x < g.width; ++x) {
            const double v = g.at(x, y);
            sum += v;
            sx += v * static_cast<double>(x);
            sy += v * static_cast<double>(y);
        }
    }
    cx = sum > 0.0 ? sx / sum : 0.0;
    cy = sum > 0.0 ? sy / sum : 0.0;
}

lenia::Project parse(const std::string& text) {
    std::istringstream in(text);
    return lenia::parse_project(in);
}

bool rejected(const std::string& text) {
    try {
        parse(text);
        return false;
    } catch (const std::invalid_argument& e) {
        std::printf("  rejected as expected: %s\n", e.what());
        return true;
    }
}

} // namespace

int main() {
    using namespace lenia;
    const GrowthParams panel;

    // --- Kernel K weights. ---------------------------------------------------------------------
    {
        KernelShape shape;
        shape.radius = 13;
        shape.core = KernelCore::Polynomial;
        shape.rings = {1.0f, -0.5f};
        expect(same(make_kernel(shape, 0.0f, 0.0f), make_kernel(shape)), "zero shift: make_kernel(shape) exactly");
        const Grid shifted = make_kernel(shape, 2.5f, -1.0f);
        expect(shifted.width == 2 * 16 + 1 && shifted.height == shifted.width, "shift 2.5 widens the grid to ceil(15.5)");

        Graph g;
        const int k = add_kernel(g, 6, KernelCore::GaussianBell, {1.0f}, KernelNormalization::Sum, 1.5f, 0.0f);
        const ConvKernel def = node_conv_kernel(g, k);
        expect(def.shape.radius == 6 && def.shape.core == KernelCore::GaussianBell && def.shape.rings.size() == 1 &&
                   def.normalization == KernelNormalization::Sum && def.shift_x == 1.5f && def.shift_y == 0.0f,
               "node_conv_kernel reads the node's parameters");
        const Grid w = make_conv_kernel(def);
        expect(std::fabs(kernel_stats(w).sum - 1.0) < 1e-5, "Sum normalization: weights sum to 1");

        // Rings past the count are stored but unused.
        const int k3 = add_kernel(g, 6, KernelCore::Step, {1.0f, 0.5f}, KernelNormalization::None);
        g.set_param(k3, conv_kernel_node::Ring3, -1.0f);
        expect(node_conv_kernel(g, k3).shape.rings == std::vector<float>{1.0f, 0.5f}, "only the first Rings weights");

        // Radius + shift over 32 cells: refused, and so is the rule using it.
        const int big = add_kernel(g, 30, KernelCore::Polynomial, {1.0f}, KernelNormalization::Sum, 5.0f, 0.0f);
        bool threw = false;
        try {
            make_conv_kernel(node_conv_kernel(g, big));
        } catch (const std::invalid_argument& e) {
            threw = true;
            std::printf("  refused as expected: %s\n", e.what());
        }
        expect(threw, "radius 30 + shift 5 exceeds the 65 x 65 limit");
        const int out = g.add_node(NodeKind::Output);
        const int d = g.add_node(NodeKind::Density);
        g.connect(add_convolve(g, d, big), 0, out, output_node::NextIn);
        const RuleCompile c = compile_rule(g, 0, panel);
        expect(c.linked && !c.error.empty() && c.program.ops.empty(), "a rule with an oversized Kernel K does not compile");
        std::printf("  oversized kernel: %s\n", c.error.c_str());
    }

    // --- Shift direction: an impulse is seen from the opposite side. ----------------------------
    {
        Grid impulse(40, 40, 0.0f);
        impulse.at(20, 20) = 1.0f;
        const auto look = [&](float sx, float sy, double& cx, double& cy) {
            ConvKernel k;
            k.shape.radius = 4;
            k.shape.core = KernelCore::GaussianBell;
            k.normalization = KernelNormalization::Sum;
            k.shift_x = sx;
            k.shift_y = sy;
            centroid(convolve(impulse, make_conv_kernel(k)), cx, cy);
        };
        double cx = 0.0, cy = 0.0;
        look(0.0f, 0.0f, cx, cy);
        expect(std::fabs(cx - 20.0) < 1e-3 && std::fabs(cy - 20.0) < 1e-3, "unshifted: U centered on the impulse");
        look(3.0f, 0.0f, cx, cy);
        expect(std::fabs(cx - 17.0) < 1e-3 && std::fabs(cy - 20.0) < 1e-3,
               "Shift X = 3: cells 3 to the left of the impulse see it (they look right)");
        look(0.0f, -2.0f, cx, cy);
        expect(std::fabs(cx - 20.0) < 1e-3 && std::fabs(cy - 22.0) < 1e-3,
               "Shift Y = -2: cells 2 below the impulse see it (they look up)");
    }

    // --- Independent kernels on Density and Nutrient. ---------------------------------------------
    {
        Graph g;
        const int d = g.add_node(NodeKind::Density);
        const int n = g.add_node(NodeKind::Nutrient);
        const int k1 = add_kernel(g, 3, KernelCore::GaussianBell, {1.0f}, KernelNormalization::Sum);
        const int k2 = add_kernel(g, 6, KernelCore::Polynomial, {1.0f, -0.5f}, KernelNormalization::None, 2.0f, -1.5f);
        const int c1 = add_convolve(g, d, k1);
        const int c2 = add_convolve(g, n, k2);
        const int out = g.add_node(NodeKind::Output);
        g.connect(c1, 0, out, output_node::NextIn);
        g.connect(c2, 0, out, output_node::NutrientIn);

        const RuleCompile c = compile_rule(g, 0, panel);
        expect(c.linked && c.error.empty(), "two Convolve nodes compile");
        expect(c.program.kernels.size() == 2, "two convolutions");
        expect(c.convolve_registers.size() == 2, "both Convolve nodes have registers");

        const WorldState world{random_grid(29, 23, 1u), random_grid(29, 23, 2u)};
        const Grid active = normalize_kernel(make_kernel(KernelShape{}), KernelNormalization::Sum);
        const WorldState next = step_rule(world, active, c.program);
        const Grid want_a = convolve(world.density, make_conv_kernel(node_conv_kernel(g, k1)));
        const Grid want_n = convolve(world.nutrient, make_conv_kernel(node_conv_kernel(g, k2)));
        expect(same(next.density, want_a), "next A = K1 ∗ A, bit for bit");
        expect(same(next.nutrient, want_n), "next N = K2 ∗ N, bit for bit (from the current N)");
        expect(same(step_graph(world, g, active, panel, 0).nutrient, want_n), "step_graph runs the same rule");

        // The single-field step has no nutrient: its Convolve reads 0.
        Graph single;
        const int sn = single.add_node(NodeKind::Nutrient);
        const int sk = add_kernel(single, 3, KernelCore::GaussianBell, {1.0f}, KernelNormalization::Sum);
        const int sout = single.add_node(NodeKind::Output);
        single.connect(add_convolve(single, sn, sk), 0, sout, output_node::NextIn);
        const Grid zero = step_rule(world.density, active, compile_rule(single, 0, panel).program);
        expect(std::all_of(zero.cells.begin(), zero.cells.end(), [](float v) { return v == 0.0f; }),
               "single-field step: a nutrient Convolve reads 0");

        // Unlinked Field: the density.
        Graph unlinked;
        const int uk = add_kernel(unlinked, 3, KernelCore::GaussianBell, {1.0f}, KernelNormalization::Sum);
        const int uout = unlinked.add_node(NodeKind::Output);
        unlinked.connect(add_convolve(unlinked, -1, uk), 0, uout, output_node::NextIn);
        expect(same(step_rule(world.density, active, compile_rule(unlinked, 0, panel).program), want_a),
               "unlinked Field convolves the density");
    }

    // --- A Convolve with the panel kernel's Kernel K is the Neighborhood U, bit for bit. --------
    {
        KernelShape shape;
        shape.radius = 13;
        shape.core = KernelCore::Polynomial;
        shape.rings = {1.0f};
        Graph lenia_rule;
        lenia_rule.add_node(NodeKind::Output);
        wire_lenia_rule(lenia_rule);
        Graph with_k = lenia_rule;
        int growth = -1;
        for (const Node& node : with_k.nodes()) {
            if (node.kind == NodeKind::Growth) growth = node.id;
        }
        int density = -1;
        for (const Node& node : with_k.nodes()) {
            if (node.kind == NodeKind::Density) density = node.id;
        }
        const int k = add_kernel(with_k, 13, KernelCore::Polynomial, {1.0f}, KernelNormalization::Sum);
        const int c = add_convolve(with_k, density, k);
        expect(growth >= 0 && with_k.connect(c, 0, growth, growth_node::UIn) == ConnectResult::Ok,
               "Convolve U relinks into Growth");
        GrowthParams g;
        g.mu = 0.15f;
        g.sigma = 0.017f;
        g.function = GrowthFunction::Polynomial;
        Grid a = random_grid(48, 48, 7u), b = a;
        for (std::uint64_t gen = 0; gen < 5; ++gen) {
            a = step_graph(a, lenia_rule, shape, g, gen);
            b = step_graph(b, with_k, shape, g, gen);
        }
        expect(same(a, b), "Kernel K = the panel kernel: same 5 generations as Neighborhood U");
    }

    // --- Sharing, reuse and limits. ----------------------------------------------------------------
    {
        Graph g;
        const int d = g.add_node(NodeKind::Density);
        const int n = g.add_node(NodeKind::Nutrient);
        const int k = add_kernel(g, 4, KernelCore::GaussianBell, {1.0f}, KernelNormalization::Sum);
        const int twin = add_kernel(g, 4, KernelCore::GaussianBell, {1.0f}, KernelNormalization::Sum);
        const int on_a = add_convolve(g, d, k);
        const int on_a_again = add_convolve(g, d, twin); // equal kernel, same field: shared
        const int on_n = add_convolve(g, n, k);          // same Kernel K, other field: its own
        const int sum = g.add_node(NodeKind::Add);
        g.connect(on_a, 0, sum, add_node::A);
        g.connect(on_a_again, 0, sum, add_node::B);
        const int out = g.add_node(NodeKind::Output);
        g.connect(sum, 0, out, output_node::NextIn);
        g.connect(on_n, 0, out, output_node::NutrientIn);
        const RuleCompile c = compile_rule(g, 0, panel);
        expect(c.error.empty() && c.program.kernels.size() == 2, "equal (field, kernel) pairs share a convolution");
        std::uint32_t r_a = kNoRegister, r_again = kNoRegister, r_n = kNoRegister;
        for (const auto& [node, reg] : c.convolve_registers) {
            if (node == on_a) r_a = reg;
            if (node == on_a_again) r_again = reg;
            if (node == on_n) r_n = reg;
        }
        expect(r_a != kNoRegister && r_a == r_again, "shared convolution: one register");
        expect(r_n != kNoRegister && r_n != r_a, "one Kernel K feeding the nutrient too: its own convolution");

        // pack_rule shifts the Convolve offsets by the kernel base; pack_rule_kernels packs them.
        const std::vector<std::uint32_t> words = pack_rule(c.program, 1000u);
        const std::vector<float> packed = pack_rule_kernels(c.program);
        expect(packed.size() == c.program.kernels[0].cells.size() + c.program.kernels[1].cells.size(),
               "pack_rule_kernels holds every weight");
        bool offsets = true;
        for (std::size_t i = 0; i < c.program.ops.size(); ++i) {
            const RuleOp& o = c.program.ops[i];
            if (o.op != static_cast<std::uint32_t>(RuleOpCode::Convolve)) continue;
            offsets = offsets && words[4 + 12 * i + 1] == o.a + 1000u &&
                      o.b * o.c == c.program.kernels[o.m0].cells.size();
            std::size_t start = 0;
            for (std::uint32_t j = 0; j < o.m0; ++j) start += c.program.kernels[j].cells.size();
            offsets = offsets && o.a == start && packed[start] == c.program.kernels[o.m0].cells[0];
        }
        expect(offsets, "Convolve ops: offset + base, size, packed in order");

        // A Convolve the rule does not use needs nothing, not even a kernel.
        add_convolve(g, d, -1);
        expect(compile_rule(g, 0, panel).error.empty(), "an unused Convolve without a kernel is harmless");

        // A used one without a kernel stops the rule, and step_graph says why.
        Graph missing;
        const int md = missing.add_node(NodeKind::Density);
        const int mout = missing.add_node(NodeKind::Output);
        missing.connect(add_convolve(missing, md, -1), 0, mout, output_node::NextIn);
        const RuleCompile mc = compile_rule(missing, 0, panel);
        expect(mc.linked && mc.error.find("Kernel K") != std::string::npos && mc.program.ops.empty(),
               "a used Convolve needs a Kernel K");
        std::printf("  missing kernel: %s\n", mc.error.c_str());
        bool threw = false;
        try {
            step_graph(random_grid(8, 8, 3u), missing, KernelShape{}, panel, 0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        expect(threw, "step_graph refuses a rule that does not compile");

        // Five different convolutions: one too many.
        Graph many;
        const int fd = many.add_node(NodeKind::Density);
        int acc = -1;
        for (int r = 1; r <= static_cast<int>(kMaxConvolutions) + 1; ++r) {
            const int conv = add_convolve(many, fd, add_kernel(many, r + 1, KernelCore::GaussianBell, {1.0f},
                                                               KernelNormalization::Sum));
            if (acc < 0) {
                acc = conv;
                continue;
            }
            const int add = many.add_node(NodeKind::Add);
            many.connect(acc, 0, add, add_node::A);
            many.connect(conv, 0, add, add_node::B);
            acc = add;
        }
        const int fout = many.add_node(NodeKind::Output);
        many.connect(acc, 0, fout, output_node::NextIn);
        const RuleCompile fc = compile_rule(many, 0, panel);
        expect(!fc.error.empty() && fc.program.kernels.empty(), "more than 4 convolutions are refused");
        std::printf("  too many: %s\n", fc.error.c_str());

        // Field and Kernel K ports only link where they belong.
        Graph types;
        const int td = types.add_node(NodeKind::Density);
        const int tk = types.add_node(NodeKind::ConvKernel);
        const int tadd = types.add_node(NodeKind::Add);
        const int tout = types.add_node(NodeKind::Output);
        expect(types.connect(td, field_node::FieldOut, tadd, add_node::A) == ConnectResult::TypeMismatch,
               "a Field is not a per-cell value");
        expect(types.connect(tk, 0, tout, output_node::KernelIn) == ConnectResult::TypeMismatch,
               "a Kernel K is not the active kernel");
        expect(types.connect(td, field_node::ValueOut, tadd, add_node::A) == ConnectResult::Ok,
               "Density's A output still links as before");
    }

    // --- The ShortLongRange demo. ------------------------------------------------------------------
    {
        const Graph g = make_preset(GraphPreset::ShortLongRange);
        const RuleCompile c = compile_rule(g, 0, panel);
        expect(c.error.empty() && c.program.kernels.size() == 2 && c.convolve_registers.size() == 2,
               "ShortLongRange: two convolutions");
        // The rule reads no Neighborhood U: a small panel kernel keeps the test quick.
        KernelShape small;
        small.radius = 2;
        Grid world = random_grid(64, 64, 11u);
        for (std::uint64_t gen = 0; gen < 40; ++gen) world = step_graph(world, g, small, panel, gen);
        std::size_t ones = 0, extremes = 0;
        bool in_range = true;
        for (const float v : world.cells) {
            in_range = in_range && std::isfinite(v) && v >= 0.0f && v <= 1.0f;
            ones += v == 1.0f ? 1u : 0u;
            extremes += v == 0.0f || v == 1.0f ? 1u : 0u;
        }
        const double n = static_cast<double>(world.cells.size());
        std::printf("  ShortLongRange after 40 generations: %.1f%% alive, %.1f%% at 0 or 1\n", 100.0 * ones / n,
                    100.0 * extremes / n);
        expect(in_range, "ShortLongRange stays in [0, 1]");
        expect(ones > 0.02 * n && ones < 0.98 * n, "ShortLongRange: noise becomes a pattern, neither empty nor full");
        expect(extremes > 0.75 * n, "ShortLongRange: most cells settle at 0 or 1");
    }

    // --- Experiment files. -------------------------------------------------------------------------
    {
        Project p = make_short_long_project();
        int k = -1;
        for (const Node& node : p.graph.nodes()) {
            if (node.kind == NodeKind::ConvKernel && k < 0) k = node.id;
        }
        p.graph.set_param(k, conv_kernel_node::ShiftX, 2.5f);
        p.graph.set_param(k, conv_kernel_node::Rings, 3.0f);
        p.graph.set_param(k, conv_kernel_node::Ring2, -0.25f);
        const std::string text = write_project(p);
        expect(text.starts_with("lenia-project 5\n"), "written as version 5");
        expect(text.find(" conv_kernel ") != std::string::npos && text.find(" convolve ") != std::string::npos,
               "both kinds are written by name");
        const Project back = parse(text);
        expect(back.graph == p.graph, "Kernel K and Convolve nodes and their links read back exactly");
        expect(write_project(back) == text, "and write back identically");

        std::string v4 = text;
        v4.replace(0, std::string("lenia-project 5").size(), "lenia-project 4");
        expect(rejected(v4), "version 4 files cannot hold Kernel K / Convolve nodes");

        // A version 4 file without them still opens (and its graph is unchanged).
        Project old;
        old.graph = make_preset(GraphPreset::NutrientTrail);
        std::string old_text = write_project(old);
        old_text.replace(0, std::string("lenia-project 5").size(), "lenia-project 4");
        expect(parse(old_text).graph == old.graph, "version 4 files without the new kinds still open");
    }

    if (failures == 0) std::printf("All Convolve node tests passed.\n");
    return failures == 0 ? 0 : 1;
}
