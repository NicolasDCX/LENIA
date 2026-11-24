// Per-cell rules (rule.hpp): the Game of Life presets move a glider one cell diagonally every
// 4 generations; the compare/select version equals the Life node; the wired Lenia presets and
// a migrated pre-rule graph run Orbium bit for bit like the built-in update; a rewired rule
// computes what its nodes say; Value nodes modulate cell inputs per generation; and oversized
// rules are refused.

#include "lenia/environment.hpp"
#include "lenia/kernel_shape.hpp"
#include "lenia/node_graph.hpp"
#include "lenia/pattern.hpp"
#include "lenia/project.hpp"
#include "lenia/rule.hpp"
#include "lenia/step.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
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

lenia::Grid shifted(const lenia::Grid& g, std::size_t dx, std::size_t dy) {
    lenia::Grid out(g.width, g.height, 0.0f);
    for (std::size_t y = 0; y < g.height; ++y) {
        for (std::size_t x = 0; x < g.width; ++x) out.at((x + dx) % g.width, (y + dy) % g.height) = g.at(x, y);
    }
    return out;
}

lenia::Grid run(lenia::Grid world, const lenia::Graph& graph, const lenia::Grid& kernel, const lenia::GrowthParams& growth,
                std::uint64_t from, std::uint64_t to) {
    for (std::uint64_t gen = from; gen < to; ++gen) world = lenia::step_graph(world, graph, kernel, growth, gen);
    return world;
}

double mass(const lenia::Grid& g) {
    double m = 0.0;
    for (float c : g.cells) m += c;
    return m;
}

// Binary world, about a third alive.
lenia::Grid random_binary(std::size_t w, std::size_t h, std::uint32_t seed) {
    lenia::Grid g(w, h);
    for (float& c : g.cells) {
        seed = seed * 1664525u + 1013904223u;
        c = (seed >> 8) % 3 == 0 ? 1.0f : 0.0f;
    }
    return g;
}

// The input `port` of `node` linked from `from`'s output 0.
void link(lenia::Graph& g, int from, int node, int port) {
    if (g.connect(from, 0, node, port) != lenia::ConnectResult::Ok) {
        std::printf("FAILED: test setup, link %d -> %d:%d\n", from, node, port);
        ++failures;
    }
}

} // namespace

int main() {
    using namespace lenia;

    // --- Port rules. ---------------------------------------------------------------------------
    {
        expect(can_link(PortType::Value, PortType::Cell), "a Value output feeds a Cell input");
        expect(!can_link(PortType::Cell, PortType::Value), "a Cell output does not feed a Value input");
        expect(!can_link(PortType::Kernel, PortType::Cell) && !can_link(PortType::Cell, PortType::Kernel),
               "Kernel and Cell ports do not mix");
        Graph g;
        const int density = g.add_node(NodeKind::Density);
        const int osc = g.add_node(NodeKind::Oscillator);
        const int add = g.add_node(NodeKind::Add);
        const int env = g.add_node(NodeKind::Environment);
        const int out = g.add_node(NodeKind::Output);
        expect(g.connect(density, 0, osc, oscillator_node::AmplitudeIn) == ConnectResult::TypeMismatch,
               "Density -> Oscillator.Amplitude rejected");
        expect(g.connect(osc, 0, add, add_node::B) == ConnectResult::Ok, "Oscillator -> Add.B accepted");
        expect(g.connect(env, environment_node::EnvironmentOut, add, add_node::A) == ConnectResult::TypeMismatch,
               "Environment (not Field) -> Add rejected");
        expect(g.connect(env, environment_node::FieldOut, add, add_node::A) == ConnectResult::Ok,
               "Environment.Field -> Add accepted");
        expect(g.connect(add, 0, out, output_node::NextIn) == ConnectResult::Ok, "Add -> Next state");
        expect(evaluate(g, 0).rule_connected, "a linked Next state is reported");
        expect(life_rule_name(8, 12) == "B3/S23" && life_rule_name(72, 12) == "B36/S23", "life rule names");
        g.set_param(g.add_node(NodeKind::Life), life_node::Birth, 7.6f);
        expect(g.nodes().back().params[life_node::Birth] == 8.0f, "Life masks round to whole numbers");
    }

    // --- Game of Life: the glider moves (+1, +1) every 4 generations. -------------------------
    const Grid moore = moore_kernel();
    {
        expect(moore.width == 3 && moore.height == 3 && kernel_stats(moore).sum == 8.0 && moore.at(1, 1) == 0.0f,
               "Moore kernel: 8 raw unit weights around a 0");
        Grid world(16, 16, 0.0f);
        stamp_pattern(world, glider_pattern().cells, 5, 5);
        expect(mass(world) == 5.0, "glider has 5 cells");
        const GrowthParams growth;
        for (GraphPreset preset : {GraphPreset::GameOfLife, GraphPreset::GameOfLifeCompare}) {
            const Graph life = make_preset(preset);
            expect(is_life_preset(preset), "is_life_preset");
            const RuleCompile rule = compile_rule(life, 0, growth);
            expect(rule.linked && rule.error.empty() && !rule.program.ops.empty(), "Life preset compiles");
            const Grid after1 = run(world, life, moore, growth, 0, 1);
            expect(after1.cells != world.cells && mass(after1) == 5.0, "the glider changes shape, keeping 5 cells");
            const Grid after4 = run(world, life, moore, growth, 0, 4);
            expect(after4.cells == shifted(world, 1, 1).cells, "glider after 4 generations = start shifted by (1, 1)");
            bool binary = true;
            for (float c : after4.cells) binary = binary && (c == 0.0f || c == 1.0f);
            expect(binary, "Life states stay exactly 0 or 1");
            // 64 generations = 16 diagonal cells: once around the 16 x 16 torus.
            expect(run(world, life, moore, growth, 0, 64).cells == world.cells, "glider wraps around the torus");
        }

        // The Life node and the compare/select graph are the same rule on any world.
        const Graph node = make_preset(GraphPreset::GameOfLife);
        const Graph wired = make_preset(GraphPreset::GameOfLifeCompare);
        const Grid soup = random_binary(40, 31, 7u);
        expect(run(soup, node, moore, growth, 0, 12).cells == run(soup, wired, moore, growth, 0, 12).cells,
               "Life node equals compare/select wiring for 12 generations of random soup");

        // A blinker oscillates with period 2; a block is still.
        Grid blinker(8, 8, 0.0f);
        blinker.at(3, 4) = blinker.at(4, 4) = blinker.at(5, 4) = 1.0f;
        const Grid b1 = run(blinker, node, moore, growth, 0, 1);
        expect(b1.at(4, 3) == 1.0f && b1.at(4, 5) == 1.0f && b1.at(3, 4) == 0.0f, "blinker turns vertical");
        expect(run(blinker, node, moore, growth, 0, 2).cells == blinker.cells, "blinker has period 2");

        // HighLife (B36/S23) differs from Conway only through births on 6.
        Graph high = make_preset(GraphPreset::GameOfLife);
        for (const Node& n : high.nodes()) {
            if (n.kind == NodeKind::Life) high.set_param(n.id, life_node::Birth, 8.0f + 64.0f);
        }
        Grid six(8, 8, 0.0f);
        const std::size_t cells[][2] = {{3, 3}, {4, 3}, {5, 3}, {3, 5}, {4, 5}, {5, 5}};
        for (const auto& c : cells) six.at(c[0], c[1]) = 1.0f;
        expect(run(six, high, moore, growth, 0, 1).at(4, 4) == 1.0f && run(six, node, moore, growth, 0, 1).at(4, 4) == 0.0f,
               "Birth mask bit 6: a dead cell with 6 neighbors is born in HighLife only");
    }

    // --- The Life experiment: its kernel is the Moore kernel, its seed the glider. -----------
    {
        const Project p = make_life_project();
        Grid raw = make_kernel(p.kernel);
        for (std::size_t i = 0; i < raw.cells.size(); ++i) raw.cells[i] += p.paint.cells[i];
        expect(normalize_kernel(raw, p.normalization).cells == moore.cells,
               "Life project: ring shape + paint, unnormalized, is the Moore kernel");
        expect(p.pattern && p.pattern->cells.cells == glider_pattern().cells.cells && p.seed == SeedKind::Pattern,
               "Life project seeds the glider");
        expect(evaluate(p.graph, 0).rule_connected && !evaluate(p.graph, 0).kernel_connected,
               "Life project: rule linked, static kernel");
        std::istringstream in(write_project(p));
        const Project r = parse_project(in);
        expect(write_project(r) == write_project(p), "Life project roundtrips");
        Grid world(128, 128, 0.0f);
        stamp_pattern(world, r.pattern->cells, 64, 64);
        expect(run(world, r.graph, raw, r.growth, 0, 4).cells == shifted(world, 1, 1).cells,
               "reopened Life project: glider after 4 generations");
    }

    // --- Lenia: wired presets and a migrated graph equal the built-in update. ----------------
    Pattern orbium;
    try {
        orbium = load_pattern_file(std::string(LENIA_PATTERN_DIR) + "/orbium.lenia");
    } catch (const std::exception& ex) {
        std::printf("FAILED: %s\n", ex.what());
        return 1;
    }
    Grid start(64, 64, 0.0f);
    stamp_pattern(start, orbium.cells, 32, 32);
    const Grid orbium_kernel = normalize_kernel(make_kernel(orbium.kernel), KernelNormalization::Sum);
    {
        const Graph idle = make_preset(GraphPreset::Idle);
        expect(evaluate(idle, 0).rule_connected, "the default (Idle) preset wires the Lenia rule visibly");
        const RuleCompile rule = compile_rule(idle, 0, orbium.growth);
        expect(rule.error.empty() && rule.program.growth != kNoRegister, "Lenia rule compiles, growth shown");
        Grid plain = start;
        for (int i = 0; i < 30; ++i) plain = step_lenia(plain, orbium_kernel, orbium.growth);
        expect(run(start, idle, orbium_kernel, orbium.growth, 0, 30).cells == plain.cells,
               "wired Lenia rule runs Orbium bit for bit like step_lenia (30 generations)");

        // A graph saved before rule nodes: environment linked into the Simulation node and on.
        Graph legacy;
        const int time = legacy.add_node(NodeKind::Time, 20, 60);
        const int env = legacy.add_node(NodeKind::Environment, 520, 230);
        const int out = legacy.add_node(NodeKind::Output, 800, 120);
        legacy.set_param(env, environment_node::Strength, 0.02f);
        legacy.set_param(time, time_node::Speed, 0.5f);
        legacy.connect(time, 0, env, environment_node::OffsetIn);
        legacy.connect(env, 0, out, output_node::EnvironmentIn);
        expect(!evaluate(legacy, 0).rule_connected, "a pre-rule graph uses the built-in update");
        Graph migrated = legacy;
        expect(wire_lenia_rule(migrated), "wire_lenia_rule migrates it");
        expect(!wire_lenia_rule(migrated), "wiring twice is refused");
        expect(evaluate(migrated, 0).rule_connected && !evaluate(migrated, 0).environment_connected,
               "migrated: rule linked, environment moved into the rule");
        const Grid a = run(start, legacy, orbium_kernel, orbium.growth, 0, 25);
        const Grid b = run(start, migrated, orbium_kernel, orbium.growth, 0, 25);
        expect(a.cells == b.cells, "migrated rule with a drifting environment equals the built-in update");
        expect(a.cells != run(start, make_preset(GraphPreset::Idle), orbium_kernel, orbium.growth, 0, 25).cells,
               "and the environment does act");

        // Panel growth flows into the rule: changing it changes the result.
        GrowthParams other = orbium.growth;
        other.mu = 0.2f;
        expect(run(start, idle, orbium_kernel, other, 0, 3).cells != run(start, idle, orbium_kernel, orbium.growth, 0, 3).cells,
               "Growth (Source = panel) follows the panel's mu");
        // The Growth node's dt output reports the panel's dt.
        for (const Node& n : idle.nodes()) {
            if (n.kind == NodeKind::Growth) {
                expect(output_value(idle, n.id, growth_node::DtOut, 0, other) == static_cast<double>(other.dt),
                       "Growth dt output = panel dt");
            }
        }
    }

    // --- A rewired rule computes what its nodes say. -----------------------------------------
    //   next = Select(U >= 0.5, Clamp(A + 0.5 * G(U; own polynomial, mu 0.25, sigma 0.05), 0, 1), A * 0.9)
    {
        Graph g;
        const int density = g.add_node(NodeKind::Density);
        const int nb = g.add_node(NodeKind::Neighborhood);
        const int growth = g.add_node(NodeKind::Growth);
        g.set_param(growth, growth_node::Source, static_cast<float>(growth_node::Own));
        g.set_param(growth, growth_node::Function, 1.0f);
        g.set_param(growth, growth_node::Mu, 0.25f);
        g.set_param(growth, growth_node::Sigma, 0.05f);
        link(g, nb, growth, growth_node::UIn);
        const int half = g.add_node(NodeKind::Multiply);
        g.set_param(half, multiply_node::B, 0.5f);
        link(g, growth, half, multiply_node::A);
        const int sum = g.add_node(NodeKind::Add);
        link(g, density, sum, add_node::A);
        link(g, half, sum, add_node::B);
        const int clamp = g.add_node(NodeKind::Clamp);
        link(g, sum, clamp, clamp_node::X);
        const int decay = g.add_node(NodeKind::Multiply);
        g.set_param(decay, multiply_node::B, 0.9f);
        link(g, density, decay, multiply_node::A);
        const int test = g.add_node(NodeKind::Compare);
        g.set_param(test, compare_node::Op, static_cast<float>(CompareOp::GreaterEqual));
        g.set_param(test, compare_node::B, 0.5f);
        link(g, nb, test, compare_node::AIn);
        const int select = g.add_node(NodeKind::Select);
        link(g, test, select, select_node::Condition);
        link(g, clamp, select, select_node::IfTrue);
        link(g, decay, select, select_node::IfFalse);
        const int out = g.add_node(NodeKind::Output);
        link(g, select, out, output_node::NextIn);

        const RuleCompile rule = compile_rule(g, 0, GrowthParams{});
        expect(rule.error.empty(), "rewired rule compiles");
        KernelShape small;
        small.radius = 4;
        const Grid k = normalize_kernel(make_kernel(small), KernelNormalization::Sum);
        Grid world(24, 20, 0.0f);
        std::uint32_t seed = 11u;
        for (float& c : world.cells) {
            seed = seed * 1664525u + 1013904223u;
            c = static_cast<float>(seed >> 8) / 16777216.0f;
        }
        const Grid u = convolve(world, k);
        const Grid next = step_graph(world, g, k, GrowthParams{}, 0);
        GrowthParams own;
        own.function = GrowthFunction::Polynomial;
        own.mu = 0.25f;
        own.sigma = 0.05f;
        bool exact = true;
        int took_true = 0;
        for (std::size_t i = 0; i < world.cells.size(); ++i) {
            const float a = world.cells[i];
            const float gu = lenia::growth(own, u.cells[i]);
            const float grown = std::min(std::max(a + gu * 0.5f, 0.0f), 1.0f);
            const bool cond = u.cells[i] >= 0.5f;
            took_true += cond ? 1 : 0;
            exact = exact && next.cells[i] == (cond ? grown : a * 0.9f);
        }
        expect(exact, "rewired rule matches its formula cell by cell");
        expect(took_true > 0 && took_true < static_cast<int>(world.cells.size()), "both Select branches are used");
        float shown = 0.0f;
        run_rule(rule.program, world.cells[0], u.cells[0], 0, 0, world.width, world.height, &shown);
        expect(shown == lenia::growth(own, u.cells[0]), "growth display is the Growth node's output");
    }

    // --- Value nodes modulate cell inputs per generation; programs pack for the GPU. ---------
    {
        Graph g;
        const int density = g.add_node(NodeKind::Density);
        const int osc = g.add_node(NodeKind::Oscillator);
        g.set_param(osc, oscillator_node::Period, 100.0f);
        g.set_param(osc, oscillator_node::Amplitude, 0.5f);
        g.set_param(osc, oscillator_node::Offset, 0.5f);
        const int clamp = g.add_node(NodeKind::Clamp);
        link(g, density, clamp, clamp_node::X);
        link(g, osc, clamp, clamp_node::High);
        const int out = g.add_node(NodeKind::Output);
        link(g, clamp, out, output_node::NextIn);
        const Grid full(4, 4, 1.0f);
        const Grid at0 = step_graph(full, g, moore, GrowthParams{}, 0);
        const Grid at25 = step_graph(full, g, moore, GrowthParams{}, 25);
        expect(std::fabs(at0.cells[0] - 0.5f) < 1e-6f && std::fabs(at25.cells[0] - 1.0f) < 1e-6f,
               "an oscillator linked into Clamp.High sets the cap per generation");
        expect(compile_rule(g, 0, {}).program != compile_rule(g, 25, {}).program, "programs differ per generation");
        expect(compile_rule(g, 7, {}).program == compile_rule(g, 7, {}).program, "and are deterministic");

        const RuleProgram prog = compile_rule(g, 0, {}).program;
        const std::vector<std::uint32_t> words = pack_rule(prog);
        expect(words.size() == 4 + 12 * prog.ops.size() && words[0] == prog.ops.size() && words[1] == prog.output,
               "pack_rule: header and 12 words per op");
        expect(words.size() <= kRuleBufferWords, "packed program fits the GPU buffer");
    }

    // --- Oversized rules are refused, not truncated. -----------------------------------------
    {
        Graph g;
        int prev = g.add_node(NodeKind::Density);
        for (int i = 0; i < 40; ++i) { // 40 Adds, each with a constant B: more than 64 ops
            const int add = g.add_node(NodeKind::Add);
            link(g, prev, add, add_node::A);
            prev = add;
        }
        const int out = g.add_node(NodeKind::Output);
        link(g, prev, out, output_node::NextIn);
        const RuleCompile rule = compile_rule(g, 0, {});
        expect(rule.linked && !rule.error.empty() && rule.program.ops.empty(), "an oversized rule reports an error");
        bool threw = false;
        try {
            step_graph(Grid(4, 4, 0.0f), g, moore, GrowthParams{}, 0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        expect(threw, "step_graph refuses an oversized rule");
    }

    // --- Unlinked Next state = no rule. -------------------------------------------------------
    {
        Graph g = make_preset(GraphPreset::GameOfLife);
        g.disconnect(g.output_node(), output_node::NextIn);
        expect(!compile_rule(g, 0, {}).linked, "unlinked Next state compiles to no rule");
    }

    std::printf(failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
