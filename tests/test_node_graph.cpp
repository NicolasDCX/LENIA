// Node graph (node_graph.hpp), kernel warps and the environment: graph editing rules,
// per-generation evaluation, presets, and that the idle graph leaves Orbium's rule untouched
// while modulated graphs change the kernel's shape and replay deterministically.

#include "lenia/environment.hpp"
#include "lenia/kernel_shape.hpp"
#include "lenia/node_graph.hpp"
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

bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

double mass(const lenia::Grid& g) {
    double m = 0.0;
    for (float c : g.cells) m += c;
    return m;
}

// Weighted mean distance from the center, sum_i |w_i| d_i / sum_i |w_i|.
double mean_radius(const lenia::Grid& k) {
    const double c = static_cast<double>(k.width / 2);
    double num = 0.0, den = 0.0;
    for (std::size_t y = 0; y < k.height; ++y) {
        for (std::size_t x = 0; x < k.width; ++x) {
            const double w = std::fabs(k.at(x, y));
            num += w * std::hypot(x - c, y - c);
            den += w;
        }
    }
    return den > 0 ? num / den : 0.0;
}

// Cells holding at least half the kernel's peak weight.
std::size_t cells_above_half(const lenia::Grid& k) {
    float peak = 0.0f;
    for (float w : k.cells) peak = std::fmax(peak, w);
    std::size_t n = 0;
    for (float w : k.cells) n += w >= 0.5f * peak ? 1 : 0;
    return n;
}

lenia::Grid run(const lenia::Grid& start, const lenia::Graph& g, const lenia::Pattern& p, std::uint64_t from,
                std::uint64_t to) {
    lenia::Grid w = start;
    for (std::uint64_t gen = from; gen < to; ++gen) w = lenia::step_graph(w, g, p.kernel, p.growth, gen);
    return w;
}

} // namespace

int main() {
    using namespace lenia;

    // --- Specs: every Value input's parameter exists. -----------------------------------------
    for (NodeKind k : node_kinds()) {
        const NodeSpec& s = node_spec(k);
        expect(s.kind == k, "spec table is indexed by kind");
        for (const PortSpec& in : s.inputs) {
            expect(in.param < static_cast<int>(s.params.size()), "input parameter index in range");
            expect(in.type == PortType::Value || in.type == PortType::Cell || in.param < 0,
                   "only Value and Cell inputs have parameters");
        }
    }

    // --- Editing: connect, type and cycle checks, replace, disconnect, remove. ----------------
    {
        Graph g;
        const int t = g.add_node(NodeKind::Time);
        const int o = g.add_node(NodeKind::Oscillator);
        const int o2 = g.add_node(NodeKind::Oscillator);
        const int k = g.add_node(NodeKind::Kernel);
        const int e = g.add_node(NodeKind::Environment);
        const int out = g.add_node(NodeKind::Output);
        expect(g.add_node(NodeKind::Output) == out, "a second Output returns the existing one");

        expect(g.connect(t, 0, o, oscillator_node::TimeIn) == ConnectResult::Ok, "Time -> Oscillator.Time");
        expect(g.link_into(o, oscillator_node::TimeIn) != nullptr, "link recorded");
        expect(g.connect(t, 0, out, output_node::KernelIn) == ConnectResult::TypeMismatch, "Value -> Kernel rejected");
        expect(g.connect(k, 0, out, output_node::EnvironmentIn) == ConnectResult::TypeMismatch,
               "Kernel -> Environment rejected");
        expect(g.connect(k, 0, out, output_node::KernelIn) == ConnectResult::Ok, "Kernel -> Output");
        expect(g.connect(e, 0, out, output_node::EnvironmentIn) == ConnectResult::Ok, "Environment -> Output");
        expect(g.connect(t, 3, o, 0) == ConnectResult::BadPort, "bad output port");
        expect(g.connect(t, 0, o, 9) == ConnectResult::BadPort, "bad input port");
        expect(g.connect(t, 0, 999, 0) == ConnectResult::BadPort, "missing node");

        expect(g.connect(o, 0, o2, oscillator_node::AmplitudeIn) == ConnectResult::Ok, "o -> o2");
        expect(g.connect(o2, 0, o, oscillator_node::OffsetIn) == ConnectResult::Cycle, "o2 -> o is a cycle");
        expect(g.connect(o, 0, o, oscillator_node::OffsetIn) == ConnectResult::Cycle, "self link is a cycle");

        expect(g.connect(o2, 0, k, kernel_node::RadiusDelta) == ConnectResult::Ok, "o2 -> Kernel.Radius");
        expect(g.connect(o, 0, k, kernel_node::RadiusDelta) == ConnectResult::Ok, "o -> Kernel.Radius replaces");
        int into_radius = 0;
        for (const Link& l : g.links()) into_radius += (l.to_node == k && l.to_port == kernel_node::RadiusDelta) ? 1 : 0;
        expect(into_radius == 1 && g.link_into(k, kernel_node::RadiusDelta)->from_node == o,
               "an input keeps only the newest link");

        expect(g.disconnect(k, kernel_node::RadiusDelta), "disconnect removes the link");
        expect(!g.disconnect(k, kernel_node::RadiusDelta), "disconnect twice is a no-op");

        const std::size_t links_before = g.links().size();
        expect(g.remove_node(o), "remove a node");
        expect(g.node(o) == nullptr, "removed node is gone");
        expect(g.links().size() == links_before - 2, "its links (Time -> o, o -> o2) are gone too");
        expect(!g.remove_node(out), "the Output node cannot be removed");

        g.set_param(k, kernel_node::RadiusDelta, 10.0f);
        expect(g.node(k)->params[kernel_node::RadiusDelta] == 3.0f, "parameters clamp to the spec range");
        g.set_param(e, environment_node::Pattern, 2.6f);
        expect(g.node(e)->params[environment_node::Pattern] == 3.0f, "enum parameters round to a choice");
        g.set_param(e, environment_node::Pattern, 99.0f);
        expect(g.node(e)->params[environment_node::Pattern] == 4.0f, "enum parameters clamp to the last choice");
    }

    // --- Values: oscillator waves, Time speed, linked inputs add to parameters. ---------------
    {
        Graph g;
        const int o = g.add_node(NodeKind::Oscillator);
        g.set_param(o, oscillator_node::Period, 100.0f);
        g.set_param(o, oscillator_node::Amplitude, 0.5f);
        g.set_param(o, oscillator_node::Offset, 0.1f);
        expect(near(output_value(g, o, 0, 0), 0.1), "sine at generation 0 is the offset");
        expect(near(output_value(g, o, 0, 25), 0.6), "sine peaks a quarter period in");
        expect(near(output_value(g, o, 0, 75), -0.4), "sine troughs three quarters in");
        expect(near(output_value(g, o, 0, 125), 0.6), "sine repeats every period");
        g.set_param(o, oscillator_node::Wave, static_cast<float>(Waveform::Triangle));
        expect(near(output_value(g, o, 0, 25), 0.6) && near(output_value(g, o, 0, 50), 0.1), "triangle wave");
        g.set_param(o, oscillator_node::Wave, static_cast<float>(Waveform::Square));
        expect(near(output_value(g, o, 0, 10), 0.6) && near(output_value(g, o, 0, 60), -0.4), "square wave");

        const int t = g.add_node(NodeKind::Time);
        g.set_param(t, time_node::Speed, 2.0f);
        g.set_param(t, time_node::Start, 5.0f);
        expect(near(output_value(g, t, 0, 10), 25.0), "Time = Start + Speed * generation");

        const int c = g.add_node(NodeKind::Constant);
        g.set_param(c, constant_node::Value, 0.25f);
        g.connect(c, 0, o, oscillator_node::OffsetIn);
        expect(near(input_value(g, o, oscillator_node::OffsetIn, 0), 0.35), "a linked input adds to its parameter");
        expect(near(input_value(g, o, oscillator_node::TimeIn, 42), 42.0), "unlinked Time is the generation");
    }

    // --- Kernel warps change the profile, not just its scale. --------------------------------
    {
        KernelShape s;
        s.radius = 13;
        s.core = KernelCore::Polynomial;
        const Grid base = make_kernel(s);
        expect(make_kernel(s, KernelWarp{}).cells == base.cells, "neutral warp reproduces make_kernel exactly");

        KernelWarp grow;
        grow.radius_delta = 0.12f;
        const Grid bigger = make_kernel(s, grow);
        expect(bigger.width == 2 * 15 + 1, "radius 13 * 1.12 = 14.56 needs a 31-wide kernel");
        expect(mean_radius(bigger) > mean_radius(base) + 0.5, "larger radius moves the weight outward");

        KernelWarp shift;
        shift.peak_shift = 0.5f;
        const Grid shifted = normalize_kernel(make_kernel(s, shift), KernelNormalization::Sum);
        const Grid base_n = normalize_kernel(base, KernelNormalization::Sum);
        expect(shifted.width == base.width, "peak shift keeps the radius");
        expect(mean_radius(shifted) > mean_radius(base_n) + 0.3, "positive peak shift moves the ring outward");
        expect(near(kernel_stats(shifted).sum, 1.0, 1e-4), "warped kernel still normalizes");

        KernelWarp sharp;
        sharp.sharpness = 1.0f;
        expect(cells_above_half(make_kernel(s, sharp)) < cells_above_half(base), "sharpness narrows the ring");
        sharp.sharpness = -1.0f;
        expect(cells_above_half(make_kernel(s, sharp)) > cells_above_half(base), "negative sharpness widens it");
    }

    // --- Environment field. --------------------------------------------------------------------
    {
        Environment env;
        env.scale = 40.0f;
        const std::size_t w = 128, h = 96, cx = w / 2, cy = h / 2;
        env.pattern = EnvironmentPattern::Uniform;
        expect(environment_field(env, 3, 7, w, h) == 1.0f, "uniform is 1");
        env.pattern = EnvironmentPattern::Stripes;
        expect(near(environment_field(env, cx, cy, w, h), 0.0), "stripes cross 0 at the center");
        expect(near(environment_field(env, cx + 10, cy, w, h), 1.0), "stripes peak a quarter wavelength along x");
        env.offset = 10.0f;
        expect(near(environment_field(env, cx + 20, cy, w, h), 1.0, 1e-4), "offset moves the stripes along the axis");
        env.offset = 0.0f;
        env.angle = 1.5707963f;
        expect(near(environment_field(env, cx, cy + 10, w, h), 1.0, 1e-4), "angle 90 deg stripes run along y");
        env.angle = 0.0f;
        env.pattern = EnvironmentPattern::Spot;
        expect(near(environment_field(env, cx, cy, w, h), 1.0) && near(environment_field(env, 0, 0, w, h), -1.0),
               "spot is +1 at its center and -1 outside");
        env.pattern = EnvironmentPattern::Gradient;
        expect(near(environment_field(env, w - 1, cy, w, h), 1.0) && near(environment_field(env, 0, cy, w, h), -1.0),
               "gradient clamps to [-1, 1]");
        env.pattern = EnvironmentPattern::Ripples;
        expect(near(environment_field(env, cx, cy, w, h), 1.0), "ripples are +1 at the center");

        bool bounded = true;
        for (int pat = 0; pat <= 4; ++pat) {
            env.pattern = static_cast<EnvironmentPattern>(pat);
            for (std::size_t y = 0; y < h; y += 5)
                for (std::size_t x = 0; x < w; x += 5) {
                    const float v = environment_field(env, x, y, w, h);
                    bounded = bounded && v >= -1.0001f && v <= 1.0001f;
                }
        }
        expect(bounded, "every pattern stays in [-1, 1]");

        // Spatial influence: a uniform world grows where E > 0 and shrinks where E < 0.
        KernelShape small;
        small.radius = 3;
        const Grid k = normalize_kernel(make_kernel(small), KernelNormalization::Sum);
        const Grid flat(32, 32, 0.5f);
        GrowthParams gp;
        Environment stripes;
        stripes.pattern = EnvironmentPattern::Stripes;
        stripes.scale = 16.0f;
        stripes.strength = 0.5f;
        const Grid next = step_lenia(flat, k, gp, stripes);
        expect(next.at(16 + 4, 5) > next.at(16 - 4, 5), "growth is biased up where E > 0, down where E < 0");
        stripes.strength = 0.0f;
        expect(step_lenia(flat, k, gp, stripes).cells == step_lenia(flat, k, gp).cells,
               "zero strength is the plain step");
    }

    // --- Presets on Orbium. ---------------------------------------------------------------------
    Pattern orbium;
    try {
        orbium = load_pattern_file(std::string(LENIA_PATTERN_DIR) + "/orbium.lenia");
    } catch (const std::exception& ex) {
        std::printf("FAILED: %s\n", ex.what());
        return 1;
    }
    Grid world(64, 64, 0.0f);
    stamp_pattern(world, orbium.cells, world.width / 2, world.height / 2);

    // The Lenia presets wire the update visibly; the environment reaches it through the
    // Environment node's Field output (test_rule.cpp checks the rule itself).
    const auto environment_of = [](const Graph& g, std::uint64_t gen) {
        for (const Node& n : g.nodes()) {
            if (n.kind == NodeKind::Environment) return node_environment(g, n.id, gen);
        }
        return Environment{};
    };
    for (GraphPreset preset : {GraphPreset::Idle, GraphPreset::BreathingKernel, GraphPreset::DriftingEnvironment}) {
        const Graph g = make_preset(preset);
        const GraphFrame f = evaluate(g, 0);
        expect(f.kernel_connected && f.kernel.neutral(), "every preset starts with a neutral, linked kernel");
        expect(f.rule_connected && !f.environment_connected, "every Lenia preset links the wired rule");
        expect(environment_of(g, 0).strength == 0.0f, "every preset starts with no environment");
    }

    // Idle: Orbium's rule, bit for bit, at any generation.
    {
        const Graph idle = make_preset(GraphPreset::Idle);
        for (std::uint64_t gen : {0ull, 1ull, 777ull, 123456ull}) {
            const GraphFrame f = evaluate(idle, gen);
            expect(f.kernel.neutral() && environment_of(idle, gen).strength == 0.0f, "idle graph stays neutral");
        }
        const Grid kernel = normalize_kernel(make_kernel(orbium.kernel), KernelNormalization::Sum);
        Grid plain = world;
        for (int i = 0; i < 20; ++i) plain = step_lenia(plain, kernel, orbium.growth);
        expect(run(world, idle, orbium, 0, 20).cells == plain.cells, "idle graph runs Orbium exactly as without it");
    }

    // Breathing kernel: the kernel's size and profile change over time.
    {
        const Graph breathing = make_preset(GraphPreset::BreathingKernel);
        const Grid k0 = graph_kernel(breathing, orbium.kernel, 0);
        // At 75 and 300 the peak-sway triangle (period 150) is at 0, the radius sine (period
        // 400) near its peak (+0.111) and at its trough (-0.12).
        const Grid k75 = graph_kernel(breathing, orbium.kernel, 75);
        const Grid k300 = graph_kernel(breathing, orbium.kernel, 300);
        expect(k75.width > k0.width && k300.width < k0.width, "kernel radius breathes");
        expect(mean_radius(k75) > mean_radius(k0) && mean_radius(k300) < mean_radius(k0),
               "kernel weight moves out and in");
        expect(near(kernel_stats(k75).sum, 1.0, 1e-4), "breathing kernel stays normalized");
        const GraphFrame f = evaluate(breathing, 37);
        expect(f.kernel.peak_shift != 0.0f, "ring peak sways too");
    }

    // Determinism: the same graph replays identically, split runs equal whole runs, and
    // modulated graphs actually change the outcome.
    {
        const Grid idle = run(world, make_preset(GraphPreset::Idle), orbium, 0, 16);
        for (GraphPreset preset : {GraphPreset::BreathingKernel, GraphPreset::DriftingEnvironment}) {
            const Graph g = make_preset(preset);
            const Grid a = run(world, g, orbium, 0, 16);
            const Grid b = run(world, g, orbium, 0, 16);
            const Grid split = run(run(world, g, orbium, 0, 7), g, orbium, 7, 16);
            expect(a.cells == b.cells, "a graph replays identically from generation 0");
            expect(a.cells == split.cells, "pausing part way (7 + 9 steps) equals 16 steps");
            expect(a.cells != idle.cells, "a modulated graph changes the simulation");
        }
    }

    std::printf(failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
