// Display graphs (display_graph.hpp): ramp sampling, mixing, field inputs, Value nodes folded
// per generation, the wired panel display matching the panel's palette lookup, the example,
// GPU packing and limits, the Display domain's node kinds, and persistence: version 3
// roundtrips the display graph bit for bit, and version 1 / 2 files open with the panel display.

#include "lenia/display_graph.hpp"
#include "lenia/node_graph.hpp"
#include "lenia/project.hpp"
#include "lenia/rule.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <random>
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

bool approx(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

template <std::size_t N> bool approx(const std::array<float, N>& a, const std::array<float, N>& b, float tol) {
    for (std::size_t i = 0; i < N; ++i) {
        if (!approx(a[i], b[i], tol)) return false;
    }
    return true;
}

bool same(float a, float b) { return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b); }

bool same(const lenia::Graph& a, const lenia::Graph& b) {
    if (a.domain() != b.domain() || a.nodes().size() != b.nodes().size() || a.links() != b.links()) return false;
    for (std::size_t i = 0; i < a.nodes().size(); ++i) {
        const lenia::Node& x = a.nodes()[i];
        const lenia::Node& y = b.nodes()[i];
        if (x.id != y.id || x.kind != y.kind || !same(x.x, y.x) || !same(x.y, y.y) || x.params.size() != y.params.size()) {
            return false;
        }
        for (std::size_t j = 0; j < x.params.size(); ++j) {
            if (!same(x.params[j], y.params[j])) return false;
        }
    }
    return true;
}

std::array<float, 3> rgb(const std::array<float, 4>& c) { return {c[0], c[1], c[2]}; }

lenia::Project parse(const std::string& text) {
    std::istringstream in(text);
    return lenia::parse_project(in);
}

void expect_rejected(const std::string& text, const char* what) {
    try {
        parse(text);
        std::printf("FAILED: accepted %s\n", what);
        ++failures;
    } catch (const std::invalid_argument& e) {
        std::printf("  rejected %-44s %s\n", what, e.what());
    } catch (const std::exception& e) {
        std::printf("FAILED: %s threw a non-invalid_argument exception: %s\n", what, e.what());
        ++failures;
    }
}

// A display graph: Fields -> Color ramp (given stops, range) -> View, for `field`.
lenia::Graph ramp_graph(const std::vector<lenia::RampStop>& stops, float lo, float hi, int field) {
    using namespace lenia;
    Graph g(GraphDomain::Display);
    const int f = g.add_node(NodeKind::Fields);
    const int r = g.add_node(NodeKind::ColorRamp);
    const int v = g.add_node(NodeKind::View);
    g.set_param(r, ramp_node::Low, lo);
    g.set_param(r, ramp_node::High, hi);
    set_ramp_stops(g, r, stops);
    g.connect(f, field, r, ramp_node::ValueIn);
    g.connect(r, 0, v, view_node::ColorIn);
    return g;
}

lenia::DisplayProgram compile(const lenia::Graph& g, std::uint64_t gen = 0) {
    const lenia::DisplayCompile c = lenia::compile_display(g, gen);
    if (!c.linked || !c.error.empty()) {
        std::printf("FAILED: display graph did not compile: %s\n", c.error.c_str());
        ++failures;
    }
    return c.program;
}

} // namespace

int main() {
    using namespace lenia;

    // --- Color ramps. --------------------------------------------------------------------------
    {
        const std::vector<RampStop> bw = {{0.0f, {0, 0, 0}}, {1.0f, {1, 1, 1}}};
        expect(approx(sample_ramp(bw, 0.25f), {0.25f, 0.25f, 0.25f}, 1e-6f), "two-stop ramp interpolates");
        expect(approx(sample_ramp(bw, -3.0f), {0, 0, 0}, 0.0f) && approx(sample_ramp(bw, 7.0f), {1, 1, 1}, 0.0f),
               "ramp holds the end colors outside its stops");
        expect(approx(sample_ramp(bw, NAN), {0, 0, 0}, 0.0f), "non-finite t samples position 0");

        // Stops inside (0.2 .. 0.6) with a hard edge (two stops at 0.4): flat ends, a jump at 0.4.
        const std::vector<RampStop> edge = {{0.2f, {1, 0, 0}}, {0.4f, {0, 1, 0}}, {0.4f, {0, 0, 1}}, {0.6f, {1, 1, 1}}};
        expect(approx(sample_ramp(edge, 0.1f), {1, 0, 0}, 0.0f), "below the first stop: its color");
        expect(approx(sample_ramp(edge, 0.3f), {0.5f, 0.5f, 0.0f}, 1e-6f), "between stops");
        expect(approx(sample_ramp(edge, 0.4f), {0, 1, 0}, 0.0f), "at a doubled stop: the first one's color");
        expect(approx(sample_ramp(edge, 0.5f), {0.5f, 0.5f, 1.0f}, 1e-6f), "after a doubled stop: the second one's");
        expect(approx(sample_ramp(edge, 0.9f), {1, 1, 1}, 0.0f), "past the last stop: its color");

        // Stops in any order: compile_display sorts them by position.
        Graph g = ramp_graph({{1.0f, {1, 1, 1}}, {0.0f, {0, 0, 0}}, {0.5f, {1, 0, 0}}}, 0.0f, 1.0f, 0);
        const DisplayProgram p = compile(g);
        expect(approx(rgb(run_display(p, 0.25f, 0, 0)), {0.5f, 0.0f, 0.0f}, 1e-6f), "unsorted stops are sorted");
        expect(p.stops.size() == 3 && p.stops[0][0] == 0.0f && p.stops[2][0] == 1.0f, "program stops sorted");

        // Low / High map the field; a reversed range flips it; an empty range is position 0.
        Graph r = ramp_graph({{0.0f, {0, 0, 0}}, {1.0f, {1, 1, 1}}}, -1.0f, 1.0f, fields_node::GrowthOut);
        expect(approx(run_display(compile(r), 0, 0, 0.5f)[0], 0.75f, 1e-6f), "Low / High map growth to t");
        r = ramp_graph({{0.0f, {0, 0, 0}}, {1.0f, {1, 1, 1}}}, 1.0f, 0.0f, fields_node::DensityOut);
        expect(approx(run_display(compile(r), 0.2f, 0, 0)[0], 0.8f, 1e-6f), "a reversed range flips the ramp");
        r = ramp_graph({{0.0f, {0, 0, 0}}, {1.0f, {1, 1, 1}}}, 0.5f, 0.5f, fields_node::DensityOut);
        expect(run_display(compile(r), 0.9f, 0, 0)[0] == 0.0f, "an empty range samples position 0");

        // Stops editing: count, clamping, padding.
        Graph s(GraphDomain::Display);
        const int id = s.add_node(NodeKind::ColorRamp);
        expect(ramp_stops(*s.node(id)).size() == 2, "a new ramp has 2 stops (black to white)");
        set_ramp_stops(s, id, {{0.3f, {2.0f, -1.0f, 0.5f}}});
        const std::vector<RampStop> got = ramp_stops(*s.node(id));
        expect(got.size() == 2 && got[0].rgb[0] == 1.0f && got[0].rgb[1] == 0.0f && got[1].position == 1.0f,
               "stops clamp to [0, 1] and pad to 2");
        std::vector<RampStop> many(14, RampStop{0.5f, {0.5f, 0.5f, 0.5f}});
        set_ramp_stops(s, id, many);
        expect(ramp_stops(*s.node(id)).size() == static_cast<std::size_t>(ramp_node::kMaxStops), "at most kMaxStops");
    }

    // --- Fields and mixing. ---------------------------------------------------------------------
    {
        Graph g(GraphDomain::Display);
        const int f = g.add_node(NodeKind::Fields);
        const int a = g.add_node(NodeKind::Color);
        const int b = g.add_node(NodeKind::Color);
        const int m = g.add_node(NodeKind::Mix);
        const int v = g.add_node(NodeKind::View);
        g.set_param(a, color_node::Red, 0.2f);
        g.set_param(a, color_node::Green, 0.4f);
        g.set_param(a, color_node::Blue, 0.8f);
        g.set_param(b, color_node::Red, 1.0f);
        g.set_param(b, color_node::Green, 0.5f);
        g.set_param(b, color_node::Blue, 0.0f);
        g.connect(a, 0, m, mix_node::AIn);
        g.connect(b, 0, m, mix_node::BIn);
        g.connect(m, 0, v, view_node::ColorIn);
        const std::array<float, 3> A{0.2f, 0.4f, 0.8f}, B{1.0f, 0.5f, 0.0f};
        const float f5 = 0.5f;
        const auto mode = [&](int md) {
            g.set_param(m, mix_node::Mode, static_cast<float>(md));
            return rgb(run_display(compile(g), 0, 0, 0));
        };
        std::array<float, 3> want{};
        for (int c = 0; c < 3; ++c) want[c] = A[c] + (B[c] - A[c]) * f5;
        expect(approx(mode(mix_node::Mix), want, 1e-6f), "Mix: A + (B - A) f");
        for (int c = 0; c < 3; ++c) want[c] = std::min(A[c] + B[c] * f5, 1.0f);
        expect(approx(mode(mix_node::Add), want, 1e-6f), "Add: A + B f, clamped");
        for (int c = 0; c < 3; ++c) want[c] = A[c] * (1.0f + (B[c] - 1.0f) * f5);
        expect(approx(mode(mix_node::Multiply), want, 1e-6f), "Multiply: A (1 + (B - 1) f)");
        for (int c = 0; c < 3; ++c) want[c] = 1.0f - (1.0f - A[c]) * (1.0f - B[c] * f5);
        expect(approx(mode(mix_node::Screen), want, 1e-6f), "Screen: 1 - (1 - A)(1 - B f)");

        // Factor from a field, per cell and clamped to [0, 1].
        g.set_param(m, mix_node::Mode, static_cast<float>(mix_node::Mix));
        g.connect(f, fields_node::NeighborhoodOut, m, mix_node::FactorIn);
        const DisplayProgram p = compile(g);
        expect(approx(rgb(run_display(p, 0, 0.0f, 0)), A, 1e-6f), "factor U = 0: A");
        expect(approx(rgb(run_display(p, 0, 1.0f, 0)), B, 1e-6f), "factor U = 1: B");
        expect(approx(rgb(run_display(p, 0, 5.0f, 0)), B, 1e-6f) && approx(rgb(run_display(p, 0, -2.0f, 0)), A, 1e-6f),
               "factor clamped to [0, 1]");
        expect(approx(rgb(run_display(p, 0, NAN, 0)), A, 1e-6f), "a NaN field factor counts as 0");

        // Unlinked colors are black; the Fields node emits each field once.
        g.disconnect(m, mix_node::AIn);
        g.disconnect(m, mix_node::FactorIn);
        g.set_param(m, mix_node::Factor, 0.0f);
        expect(approx(rgb(run_display(compile(g), 0, 0, 0)), {0, 0, 0}, 0.0f), "unlinked A is black");
        const int r1 = g.add_node(NodeKind::ColorRamp);
        const int r2 = g.add_node(NodeKind::ColorRamp);
        g.connect(f, fields_node::DensityOut, r1, ramp_node::ValueIn);
        g.connect(f, fields_node::DensityOut, r2, ramp_node::ValueIn);
        g.connect(r1, 0, m, mix_node::AIn);
        g.connect(r2, 0, m, mix_node::BIn);
        int field_ops = 0;
        for (const DisplayOp& op : compile(g).ops) field_ops += op.op == static_cast<std::uint32_t>(DisplayOpCode::Field);
        expect(field_ops == 1, "a field read by two ramps is read once");
    }

    // --- Value nodes are folded per generation. -------------------------------------------------
    {
        Graph g = make_display_preset(DisplayPreset::GrowthOverlay);
        const DisplayCompile c0 = compile_display(g, 0);
        const DisplayCompile c60 = compile_display(g, 60); // a quarter period: factor 0.75
        expect(c0.linked && c0.error.empty(), "the example compiles");
        expect(c0.program.ops.size() <= kMaxDisplayOps && c0.program.stops.size() <= kMaxDisplayStops,
               "the example fits the GPU limits");
        expect(!(c0.program == c60.program), "the example's pulse changes the program over time");
        expect(compile_display(g, 60).program == c60.program, "the same generation compiles the same program");
        bool in_range = true;
        std::mt19937 rng(3);
        std::uniform_real_distribution<float> u(-1.5f, 1.5f);
        for (int i = 0; i < 200; ++i) {
            const auto c = run_display(c0.program, u(rng), u(rng), u(rng));
            for (float x : c) in_range = in_range && x >= 0.0f && x <= 1.0f;
        }
        expect(in_range, "the example's colors stay within [0, 1]");
        // Where growth is 0 the growth ramp is black, so Screen leaves the density color.
        const Node* density_ramp = nullptr;
        for (const Node& n : g.nodes()) {
            if (n.kind == NodeKind::ColorRamp && !density_ramp) density_ramp = &n;
        }
        std::vector<RampStop> sorted = ramp_stops(*density_ramp);
        expect(approx(rgb(run_display(c0.program, 0.7f, 0.0f, 0.0f)), sample_ramp(sorted, 0.7f), 1e-6f),
               "example: zero growth shows the density ramp");
    }

    // --- The wired panel display reproduces the panel's palette lookup. -------------------------
    for (int pal = 0; pal < kDisplayPaletteCount; ++pal) {
        for (int field = 0; field < kDisplayFieldCount; ++field) {
            const DisplayRange range{static_cast<DisplayPalette>(pal), field == 2 ? -1.0f : 0.0f,
                                     field == 1 ? 0.3f : 1.0f};
            Graph g = make_display_preset(DisplayPreset::WiredPanel, static_cast<DisplayField>(field), range);
            const DisplayProgram p = compile(g);
            const auto at = [&](float v) {
                return rgb(run_display(p, field == 0 ? v : 0.0f, field == 1 ? v : 0.0f, field == 2 ? v : 0.0f));
            };
            bool exact = true, close = true;
            for (int i = 0; i < kPaletteTableSize; ++i) {
                // Values on the lookup table's entries match to rounding; in between, within
                // the table's 1/255 quantization.
                const float v = range.lo + (range.hi - range.lo) * static_cast<float>(i) / (kPaletteTableSize - 1);
                exact = exact && approx(at(v), panel_color(range, v), 2e-5f);
                const float w = v + (range.hi - range.lo) * 0.4f / (kPaletteTableSize - 1);
                close = close && approx(at(w), panel_color(range, w), 0.02f);
            }
            expect(exact, "wired panel display = panel lookup on table entries");
            expect(close, "wired panel display = panel lookup within quantization");
        }
    }
    {
        Graph g = make_display_preset(DisplayPreset::Panel);
        expect(!compile_display(g, 0).linked, "the Panel preset leaves View unlinked");
        expect(wire_panel_display(g, DisplayField::Growth, {DisplayPalette::RedBlue, -1.0f, 1.0f}), "wire the panel");
        expect(compile_display(g, 0).linked, "wired: linked");
        expect(!wire_panel_display(g, DisplayField::Density, {}), "wiring twice changes nothing");
    }

    // --- GPU packing and limits. ----------------------------------------------------------------
    {
        const DisplayProgram p = compile(make_display_preset(DisplayPreset::GrowthOverlay));
        const std::vector<std::uint32_t> w = pack_display(p);
        expect(w.size() == kDisplayBufferWords, "packed size is fixed");
        expect(w[0] == p.ops.size() && w[1] == p.output && w[2] == p.stops.size(), "packed header");
        expect(w[4 + 12 * 1] == p.ops[1].op && std::bit_cast<float>(w[4 + 12 * kMaxDisplayOps]) == p.stops[0][0],
               "packed ops and stops at their offsets");

        Graph big(GraphDomain::Display);
        const int view = big.add_node(NodeKind::View);
        int prev = big.add_node(NodeKind::Color);
        for (int i = 0; i < 40; ++i) {
            const int m = big.add_node(NodeKind::Mix);
            big.connect(prev, 0, m, mix_node::AIn);
            prev = m;
        }
        big.connect(prev, 0, view, view_node::ColorIn);
        const DisplayCompile c = compile_display(big, 0);
        expect(c.linked && !c.error.empty() && c.program.ops.empty(), "an oversized display graph is refused");
    }

    // --- Domains: each graph takes its own node kinds. ------------------------------------------
    {
        Graph d(GraphDomain::Display);
        expect(d.add_node(NodeKind::Density) == -1 && d.add_node(NodeKind::Output) == -1,
               "rule and Simulation nodes are refused in a display graph");
        expect(d.add_node(NodeKind::Oscillator) > 0, "Value nodes are allowed in a display graph");
        const int v = d.add_node(NodeKind::View);
        expect(d.add_node(NodeKind::View) == v && d.output_node() == v, "one View node, the sink");
        expect(!d.remove_node(v), "the View node cannot be removed");
        Graph s;
        expect(s.add_node(NodeKind::ColorRamp) == -1, "display nodes are refused in a simulation graph");
        expect(!evaluate(d, 0).rule_connected && !wire_lenia_rule(d), "simulation evaluation ignores display graphs");

        std::vector<Node> nodes = d.nodes();
        Node rule;
        rule.id = 99;
        rule.kind = NodeKind::Density;
        nodes.push_back(rule);
        expect(!d.assign(nodes, {}).empty(), "assign rejects a rule node in a display graph");
        nodes.pop_back();
        nodes.erase(nodes.begin() + 1); // the View
        expect(!d.assign(nodes, {}).empty(), "assign needs a View node");
        Graph sim = make_preset(GraphPreset::Idle);
        std::vector<Node> sim_nodes = sim.nodes();
        Node ramp;
        ramp.id = 999;
        ramp.kind = NodeKind::ColorRamp;
        for (const ParamSpec& ps : node_spec(NodeKind::ColorRamp).params) ramp.params.push_back(ps.def);
        sim_nodes.push_back(ramp);
        expect(!sim.assign(sim_nodes, sim.links()).empty(), "assign rejects a display node in a simulation graph");
        expect(compile_rule(sim, 0, {}).linked, "the simulation graph is unchanged by the refused assign");
    }

    // --- Persistence. ---------------------------------------------------------------------------
    {
        Project p;
        p.graph = make_preset(GraphPreset::BreathingKernel);
        p.display_graph = make_display_preset(DisplayPreset::GrowthOverlay);
        const int extra = p.display_graph.add_node(NodeKind::ColorRamp, -12.5f, 3.25f);
        set_ramp_stops(p.display_graph, extra, palette_ramp(DisplayPalette::Inferno)); // 10 stops
        const std::string text = write_project(p);
        try {
            const Project q = parse(text);
            expect(same(q.display_graph, p.display_graph), "display graph roundtrips bit for bit");
            expect(same(q.graph, p.graph), "the simulation graph is kept beside it");
            expect(write_project(q) == text, "write(parse(write(p))) == write(p)");
            expect(compile_display(q.display_graph, 77).program == compile_display(p.display_graph, 77).program,
                   "reopened display graph compiles the same program");
        } catch (const std::exception& e) {
            std::printf("FAILED: roundtrip: %s\n%s\n", e.what(), text.c_str());
            ++failures;
        }

        // Version 2 (and 1) files: no [display_graph]; they open with the panel display.
        Project old;
        old.graph = make_preset(GraphPreset::Idle);
        std::string v2 = write_project(old);
        v2.replace(0, v2.find('\n'), "lenia-project 2");
        const std::size_t start = v2.find("[display_graph]");
        const std::size_t end = v2.find("[graph]");
        v2.erase(start, end - start);
        try {
            const Project q = parse(v2);
            expect(q.display_graph.domain() == GraphDomain::Display && q.display_graph.nodes().size() == 1 &&
                       q.display_graph.nodes()[0].kind == NodeKind::View,
                   "a version 2 file gets a lone View node");
            expect(!compile_display(q.display_graph, 0).linked, "so it shows the panel display, as when saved");
            expect(same(q.graph, old.graph), "its simulation graph reads as before");
            expect(write_project(q).rfind("lenia-project 3\n", 0) == 0, "and saves as version 3");
        } catch (const std::exception& e) {
            std::printf("FAILED: version 2 file: %s\n%s\n", e.what(), v2.c_str());
            ++failures;
        }

        std::printf("Invalid files:\n");
        const std::string base = write_project(old);
        expect_rejected(v2 + "[display_graph]\nnode 1 view 0 0\n", "[display_graph] in a version 2 file");
        {
            std::string missing = base;
            const std::size_t s = missing.find("[display_graph]");
            missing.erase(s, missing.find("[graph]") - s);
            expect_rejected(missing, "a version 3 file without [display_graph]");
        }
        const auto in_display = [&](const std::string& lines) {
            std::string t = base;
            t.insert(t.find("[graph]"), lines);
            return t;
        };
        expect_rejected(in_display("node 2 density 0 0\n"), "a rule node in [display_graph]");
        expect_rejected(in_display("node 2 view 0 0\n"), "two View nodes");
        expect_rejected(in_display("node 2 color 0 0 1 1\n"), "a Color node missing a parameter");
        expect_rejected(in_display("node 2 color 0 0 1 1 1.5\n"), "a color component above 1");
        expect_rejected(in_display("node 2 mix 0 0 1.5 0.5\n"), "a fractional mix mode");
        expect_rejected(in_display("node 2 color 0 0 1 1 1\nlink 2 0 1 0\nnode 3 color 0 0 1 1 1\nlink 3 0 1 0\n"),
                        "two links into the View");
        expect_rejected(in_display("node 2 fields 0 0\nlink 2 0 1 0\n"), "a field linked straight into the View");
        expect_rejected(base + "node 99 ramp 0 0\n", "a display node in [graph]");
    }

    if (failures == 0) std::printf("All display graph tests passed.\n");
    return failures == 0 ? 0 : 1;
}
