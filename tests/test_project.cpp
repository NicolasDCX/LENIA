// Experiment files (project.hpp): a project survives write -> parse bit for bit, including the
// full node graph (ids, positions, parameters, links), the kernel paint layer and the seed
// pattern; a reopened graph replays the simulation identically; a hand-written version 1 file
// keeps reading; and malformed or invalid files are rejected with an exception.

#include "lenia/node_graph.hpp"
#include "lenia/pattern.hpp"
#include "lenia/project.hpp"
#include "lenia/step.hpp"

#include <bit>
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

bool same(float a, float b) { return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b); }

bool same(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!same(a[i], b[i])) return false;
    }
    return true;
}

bool same(const lenia::Grid& a, const lenia::Grid& b) {
    return a.width == b.width && a.height == b.height && same(a.cells, b.cells);
}

bool same(const lenia::Graph& a, const lenia::Graph& b) {
    if (a.nodes().size() != b.nodes().size() || a.links() != b.links()) return false;
    for (std::size_t i = 0; i < a.nodes().size(); ++i) {
        const lenia::Node& x = a.nodes()[i];
        const lenia::Node& y = b.nodes()[i];
        if (x.id != y.id || x.kind != y.kind || !same(x.x, y.x) || !same(x.y, y.y) || !same(x.params, y.params)) {
            return false;
        }
    }
    return true;
}

lenia::Project parse(const std::string& text) {
    std::istringstream in(text);
    return lenia::parse_project(in);
}

// Expects parse_project to reject `text`; prints the message it gave so the log shows why.
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

std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    std::istringstream in(text);
    std::string l;
    while (std::getline(in, l)) lines.push_back(l);
    return lines;
}

std::string join(const std::vector<std::string>& lines) {
    std::string s;
    for (const std::string& l : lines) s += l + "\n";
    return s;
}

// First line starting with `prefix` replaced by `with` (removed when `with` is empty).
std::string replace_line(const std::string& text, const std::string& prefix, const std::string& with) {
    std::vector<std::string> lines = split_lines(text);
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (lines[i].rfind(prefix, 0) == 0) {
            if (with.empty()) lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(i));
            else lines[i] = with;
            return join(lines);
        }
    }
    std::printf("FAILED: test setup, no line starts with '%s'\n", prefix.c_str());
    ++failures;
    return text;
}

// Removes the section `name` (header and body).
std::string remove_section(const std::string& text, const std::string& name) {
    std::vector<std::string> out;
    bool skipping = false;
    for (const std::string& l : split_lines(text)) {
        if (!l.empty() && l[0] == '[') skipping = l == "[" + name + "]";
        if (!skipping) out.push_back(l);
    }
    return join(out);
}

// A small project touching every field: 5×5 paint, a 3×2 pattern, the idle graph
// (1 time, 2 oscillator, 3 kernel, 4 environment, 5 output, then its wired Lenia rule from 6).
lenia::Project small_project() {
    lenia::Project p;
    p.kernel.radius = 2;
    p.kernel.core = lenia::KernelCore::Step;
    p.kernel.rings = {1.0f, -0.5f};
    p.paint = lenia::Grid(5, 5, 0.0f);
    p.paint.at(1, 2) = 0.25f;
    p.pattern = lenia::Pattern{};
    p.pattern->name = "Tiny";
    p.pattern->cells = lenia::Grid(3, 2, 0.5f);
    p.graph = lenia::make_preset(lenia::GraphPreset::Idle);
    return p;
}

} // namespace

int main() {
    using namespace lenia;

    // --- Full roundtrip: every value bit for bit. --------------------------------------------
    Pattern orbium;
    try {
        orbium = load_pattern_file(std::string(LENIA_PATTERN_DIR) + "/orbium.lenia");
    } catch (const std::exception& ex) {
        std::printf("FAILED: %s\n", ex.what());
        return 1;
    }

    Project p;
    p.kernel.radius = 9;
    p.kernel.core = KernelCore::Polynomial;
    p.kernel.rings = {0.5f, 1.0f, -0.333333343f};
    p.normalization = KernelNormalization::Positive;
    p.paint = Grid(19, 19, 0.0f);
    p.paint.at(3, 4) = 0.1f;
    p.paint.at(9, 9) = -1.3f;
    p.paint.at(18, 0) = 1e-7f;
    p.paint.at(0, 18) = -0.0f;
    p.paint.at(10, 2) = 2.0f;
    p.growth = {0.123456789f, 0.0170000009f, 0.1f, GrowthFunction::Polynomial};
    p.world_size = 512;
    p.seed = SeedKind::Random;
    p.random_seed = 12345;
    p.pattern = orbium;
    p.pattern->source = "  surrounding spaces and # hash kept  ";
    p.field = DisplayField::Growth;
    p.display[0] = {DisplayPalette::IceFire, -0.25f, 0.75f};
    p.display[1] = {DisplayPalette::Grayscale, 0.0f, 0.3f};
    p.display[2] = {DisplayPalette::RedBlue, -1.0f, 1.0f};
    p.graph = make_preset(GraphPreset::BreathingKernel);
    const int c = p.graph.add_node(NodeKind::Constant, 123.456f, -7.25f);
    p.graph.set_param(c, constant_node::Value, 0.0375f);
    p.graph.remove_node(1); // Time: ids no longer contiguous, and its links go too
    const int env = p.graph.add_node(NodeKind::Environment, 600.5f, 400.0f);
    p.graph.set_param(env, environment_node::Pattern, 4.0f);
    p.graph.connect(c, 0, env, environment_node::StrengthIn);

    const std::string text = write_project(p);
    Project q;
    try {
        q = parse(text);
    } catch (const std::exception& ex) {
        std::printf("FAILED: roundtrip parse: %s\n%s\n", ex.what(), text.c_str());
        return 1;
    }
    expect(q.kernel.radius == 9 && q.kernel.core == KernelCore::Polynomial, "kernel radius and core");
    expect(same(q.kernel.rings, p.kernel.rings), "ring weights bit for bit");
    expect(q.normalization == KernelNormalization::Positive, "normalization");
    expect(same(q.paint, p.paint), "paint layer bit for bit (including -0 and 1e-7)");
    expect(same(q.growth.mu, p.growth.mu) && same(q.growth.sigma, p.growth.sigma) && same(q.growth.dt, p.growth.dt) &&
               q.growth.function == GrowthFunction::Polynomial,
           "growth");
    expect(q.world_size == 512 && q.seed == SeedKind::Random && q.random_seed == 12345, "world settings");
    expect(q.pattern.has_value(), "pattern present");
    if (q.pattern) {
        expect(q.pattern->name == orbium.name && q.pattern->license == orbium.license, "pattern name and license");
        expect(q.pattern->source == "surrounding spaces and # hash kept", "pattern source trimmed, '#' kept");
        expect(same(q.pattern->cells, orbium.cells), "pattern cells bit for bit");
    }
    expect(q.field == DisplayField::Growth, "display field");
    bool display_same = true;
    for (int i = 0; i < kDisplayFieldCount; ++i) {
        display_same = display_same && q.display[i].palette == p.display[i].palette &&
                       same(q.display[i].lo, p.display[i].lo) && same(q.display[i].hi, p.display[i].hi);
    }
    expect(display_same, "display palettes and ranges");
    expect(same(q.graph, p.graph), "graph: ids, kinds, positions, parameters and links");
    expect(write_project(q) == text, "write(parse(write(p))) == write(p)");

    // The reopened graph still edits like the original: new ids continue past the largest.
    {
        Graph a = p.graph, b = q.graph;
        expect(a.add_node(NodeKind::Time) == b.add_node(NodeKind::Time), "next node id restored");
    }

    // --- A reopened experiment replays identically from generation 0. ------------------------
    {
        Project o;
        o.kernel = orbium.kernel;
        o.growth = orbium.growth;
        o.graph = make_preset(GraphPreset::BreathingKernel);
        o.graph.set_param(o.graph.node(4)->id, environment_node::Strength, 0.01f); // environment on, too
        const Project r = parse(write_project(o));
        Grid start(64, 64, 0.0f);
        stamp_pattern(start, orbium.cells, 32, 32);
        Grid a = start, b = start;
        for (std::uint64_t gen = 0; gen < 40; ++gen) {
            a = step_graph(a, o.graph, o.kernel, o.growth, gen);
            b = step_graph(b, r.graph, r.kernel, r.growth, gen);
        }
        expect(same(a, b), "40 generations from the reopened project match bit for bit");
    }

    // --- Optional parts: no paint, no pattern. -----------------------------------------------
    {
        Project e;
        e.graph = make_preset(GraphPreset::Idle);
        const Project r = parse(write_project(e));
        expect(r.paint.cells.empty() && !r.pattern, "no paint, no pattern roundtrip as none");
        expect(write_project(r) == write_project(e), "minimal project roundtrip");
    }

    // --- Version 1 as written by hand stays readable. ----------------------------------------
    {
        const char* v1 = "lenia-project 1\n"
                         "# comment\n"
                         "[world]\nsize 128\nseed impulse\nrandom_seed 7\n"
                         "[kernel]\nradius 1\ncore gaussian\nrings 1\nnormalization none\npaint 3 3\n"
                         "0 0 0\n0 0.5 0\n0 0 -0.5\n"
                         "[growth]\nfunction gaussian\nmu 0.2\nsigma 0.02\ndt 0.05\n"
                         "[display]\nfield neighborhood\ndensity viridis 0 1\nneighborhood inferno 0 0.3\n"
                         "growth redblue -1 1\n"
                         "[graph]\nnode 7 output 10 20\nnode 3 constant 0 0 0.25\n";
        try {
            const Project r = parse(v1);
            expect(r.world_size == 128 && r.seed == SeedKind::Impulse && r.random_seed == 7, "v1: world");
            expect(r.kernel.core == KernelCore::GaussianBell && r.normalization == KernelNormalization::None,
                   "v1: kernel");
            expect(r.paint.width == 3 && same(r.paint.at(2, 2), -0.5f), "v1: paint rows top first");
            expect(r.field == DisplayField::Neighborhood, "v1: display field");
            expect(r.graph.nodes().size() == 2 && r.graph.output_node() == 7 && r.graph.links().empty(),
                   "v1: graph without links");
        } catch (const std::exception& ex) {
            std::printf("FAILED: hand-written v1 file: %s\n", ex.what());
            ++failures;
        }

        // A version 1 graph (environment linked into the Simulation node) opens as it was saved,
        // runs the built-in update, and can be migrated to a wired rule with the same result.
        const std::string v1_graph = std::string(v1).substr(0, std::string(v1).find("[graph]")) +
                                     "[graph]\nnode 1 time 20 60 0.5 0\nnode 2 environment 520 230 3 0.02 64 0 0\n"
                                     "node 5 output 800 120\nlink 1 0 2 1\nlink 2 0 5 1\n";
        try {
            Project r = parse(v1_graph);
            expect(!evaluate(r.graph, 0).rule_connected && evaluate(r.graph, 0).environment_connected,
                   "v1 graph: built-in update with its environment");
            Graph migrated = r.graph;
            expect(wire_lenia_rule(migrated), "v1 graph: migrates to a wired rule");
            Grid start(32, 32, 0.0f);
            stamp_pattern(start, orbium.cells, 16, 16);
            KernelShape small = orbium.kernel;
            small.radius = 5;
            Grid a = start, b = start;
            for (std::uint64_t gen = 0; gen < 10; ++gen) {
                a = step_graph(a, r.graph, small, orbium.growth, gen);
                b = step_graph(b, migrated, small, orbium.growth, gen);
            }
            expect(same(a, b), "v1 graph: migrated rule replays the built-in update bit for bit");
            r.graph = migrated;
            expect(write_project(r).rfind("lenia-project " + std::to_string(kProjectVersion) + "\n", 0) == 0,
                   "a migrated project saves as the current version");
        } catch (const std::exception& ex) {
            std::printf("FAILED: v1 graph: %s\n", ex.what());
            ++failures;
        }
        expect_rejected(std::string(v1) + "node 9 density 0 0\n", "a rule node in a version 1 file");
        expect_rejected(replace_line(v1_graph, "lenia-project", "lenia-project 1") + "node 9 constant 0 0 1\nlink 9 0 5 2\n",
                        "a Next state link in a version 1 file");
        expect(parse(replace_line(std::string(v1), "lenia-project", "lenia-project 2") + "node 9 density 0 0\n")
                       .graph.nodes()
                       .size() == 3,
               "the same rule node in a version 2 file");
    }

    // --- Graph::assign leaves the graph unchanged when it refuses. ---------------------------
    {
        Graph g = make_preset(GraphPreset::DriftingEnvironment);
        const Graph before = g;
        std::vector<Node> nodes = g.nodes();
        nodes.push_back(nodes.back()); // duplicate id
        expect(!g.assign(nodes, g.links()).empty(), "assign rejects a duplicate id");
        expect(same(g, before), "graph unchanged after a rejected assign");
        expect(g.assign(before.nodes(), before.links()).empty() && same(g, before), "assign accepts a valid graph");
    }

    // --- Invalid files. -----------------------------------------------------------------------
    const std::string base = write_project(small_project());
    try {
        parse(base);
    } catch (const std::exception& ex) {
        std::printf("FAILED: base project for the invalid-file cases: %s\n%s\n", ex.what(), base.c_str());
        return 1;
    }
    std::printf("Invalid files:\n");
    expect_rejected("", "an empty file");
    expect_rejected("# only a comment\n", "a file with only comments");
    expect_rejected(replace_line(base, "lenia-project", "lenia-pattern 1"), "a wrong header");
    expect_rejected(base.substr(base.find('\n') + 1), "a missing header");
    expect_rejected(replace_line(base, "lenia-project", "lenia-project " + std::to_string(kProjectVersion + 1)),
                    "a newer format version");
    expect_rejected(replace_line(base, "lenia-project", "lenia-project 0"), "format version 0");
    expect_rejected(replace_line(base, "lenia-project", "lenia-project one"), "a non-numeric version");
    expect_rejected(remove_section(base, "growth"), "a missing [growth]");
    expect_rejected(remove_section(base, "graph"), "a missing [graph]");
    expect_rejected(base + "[world]\nsize 256\nseed random\nrandom_seed 1\n", "a duplicate section");
    expect_rejected(base + "[extras]\n", "an unknown section");
    expect_rejected(replace_line(base, "radius", "radius 2\ncolor blue"), "an unknown key");
    expect_rejected(replace_line(base, "radius", "radius 2\nradius 2"), "a duplicate key");
    expect_rejected(replace_line(base, "dt ", ""), "a missing key");
    expect_rejected(replace_line(base, "radius", "radius 2 3"), "an extra value");
    expect_rejected(replace_line(base, "radius", "radius 0"), "radius 0");
    expect_rejected(replace_line(base, "radius", "radius 2.5"), "a fractional radius");
    expect_rejected(replace_line(base, "core", "core hexagonal"), "an unknown core");
    expect_rejected(replace_line(base, "rings", "rings"), "no ring weights");
    expect_rejected(replace_line(base, "normalization", "normalization max"), "an unknown normalization");
    expect_rejected(replace_line(base, "mu ", "mu abc"), "a malformed number");
    expect_rejected(replace_line(base, "mu ", "mu 0.15x"), "trailing characters on a number");
    expect_rejected(replace_line(base, "mu ", "mu nan"), "mu nan");
    expect_rejected(replace_line(base, "mu ", "mu inf"), "mu inf");
    expect_rejected(replace_line(base, "mu ", "mu 1e999"), "mu out of float range");
    expect_rejected(replace_line(base, "sigma ", "sigma 0"), "sigma 0");
    expect_rejected(replace_line(base, "dt ", "dt -0.1"), "a negative dt");
    expect_rejected(replace_line(base, "function", "function sigmoid"), "an unknown growth function");
    expect_rejected(replace_line(base, "size", "size 0"), "world size 0");
    expect_rejected(replace_line(base, "seed ", "seed glider"), "an unknown seed");
    expect_rejected(replace_line(base, "random_seed", "random_seed -1"), "a negative random seed");
    expect_rejected(replace_line(base, "paint 5 5", "paint 7 7"), "paint rows missing");
    expect_rejected(replace_line(base, "paint 5 5", "paint 4 5"), "a paint row of the wrong length");
    expect_rejected(replace_line(base, "paint 5 5", "paint none\n0 0 0 0 0"), "stray numbers after paint none");
    expect_rejected(replace_line(base, "radius", "radius 3"), "paint size not matching the radius");
    expect_rejected(replace_line(base, "0 0.25 0 0 0", "0 2.5 0 0 0"), "a paint value beyond [-2, 2]");
    expect_rejected(replace_line(base, "0 0.25 0 0 0", "0 nan 0 0 0"), "a nan paint value");
    expect_rejected(replace_line(base, "cells 3 2", "cells 3 3"), "pattern cells rows missing");
    expect_rejected(replace_line(base, "cells 3 2", "cells none"), "pattern without cells");
    expect_rejected(replace_line(base, "field", "field temperature"), "an unknown display field");
    expect_rejected(replace_line(base, "density", "density rainbow 0 1"), "an unknown palette");
    expect_rejected(replace_line(base, "density", "density viridis 0"), "a display range missing high");
    expect_rejected(base + "node 99 output 0 0\n", "two output nodes");
    expect_rejected(replace_line(base, "node 5 output", ""), "no output node");
    expect_rejected(base + "node 1 constant 0 0 0.1\n", "a duplicate node id");
    expect_rejected(base + "node 0 constant 0 0 0.1\n", "node id 0");
    expect_rejected(base + "node 99 wormhole 0 0\n", "an unknown node kind");
    expect_rejected(base + "node 99 constant 0 0\n", "too few parameters");
    expect_rejected(base + "node 99 constant 0 0 1 2\n", "too many parameters");
    expect_rejected(base + "node 99 constant 0 0 2e6\n", "a parameter out of range");
    expect_rejected(base + "node 99 oscillator 0 0 1.5 240 0 0.1 0\n", "a fractional enum parameter");
    expect_rejected(base + "node 99 life 0 0 8.5 12\n", "a fractional Life mask");
    expect_rejected(base + "node 99 constant nan 0 0.1\n", "a nan node position");
    expect_rejected(base + "link 1 0 42 0\n", "a link to a missing node");
    expect_rejected(base + "link 1 0 2 7\n", "a link to a missing port");
    expect_rejected(base + "link 3 0 2 1\n", "a link between port types");
    expect_rejected(base + "node 99 density 0 0\nlink 99 0 2 1\n", "a Cell output into a Value input");
    expect_rejected(base + "node 99 oscillator 0 0 0 240 0 0.1 0\nlink 99 0 2 1\nlink 2 0 99 1\n", "a cycle");
    expect_rejected(base + "link 1 0 2 0\n", "two links into one input");
    expect_rejected(base + "edge 1 0 2 0\n", "an unknown key in [graph]");
    expect_rejected(base.substr(0, base.find("[graph]")) + "[graph]\nnode 5 output 0", "a truncated file");

    if (failures == 0) std::printf("All project tests passed.\n");
    return failures == 0 ? 0 : 1;
}
