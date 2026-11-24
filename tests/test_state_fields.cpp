// Two state fields (world.hpp): Density A and Nutrient N.
//
// - seed_nutrient starts the nutrient deterministically (uniform, random, gradient);
// - a rule writes both fields from the same current state (a swap rule exchanges them exactly);
// - Next nutrient unlinked keeps the nutrient bit for bit, and the density runs exactly as
//   with one field; Next nutrient without Next state is refused;
// - the NutrientTrail demo starts exactly as the wired Lenia rule, depletes the nutrient only
//   where the density has been, lets it recover once the density is gone, and keeps Orbium alive;
// - experiments and snapshots store the nutrient (versions 4 and 2) and read older files.

#include "lenia/display_graph.hpp"
#include "lenia/kernel_shape.hpp"
#include "lenia/node_graph.hpp"
#include "lenia/pattern.hpp"
#include "lenia/project.hpp"
#include "lenia/rule.hpp"
#include "lenia/snapshot.hpp"
#include "lenia/world.hpp"

#include <algorithm>
#include <bit>
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

// The first line starting with `prefix` replaced by `with`.
std::string replace_line(const std::string& text, const std::string& prefix, const std::string& with) {
    std::istringstream in(text);
    std::string out, l;
    bool done = false;
    while (std::getline(in, l)) {
        if (!done && l.rfind(prefix, 0) == 0) {
            out += with + "\n";
            done = true;
        } else {
            out += l + "\n";
        }
    }
    if (!done) {
        std::printf("FAILED: test setup, no line starts with '%s'\n", prefix.c_str());
        ++failures;
    }
    return out;
}

// The NutrientTrail preset's uptake Multiply (B = −kTrailUptake).
int uptake_node(const lenia::Graph& g) {
    for (const lenia::Node& n : g.nodes()) {
        if (n.kind == lenia::NodeKind::Multiply && n.params[lenia::multiply_node::B] == -lenia::kTrailUptake) return n.id;
    }
    return -1;
}

} // namespace

int main() {
    using namespace lenia;

    // --- Starting nutrient. --------------------------------------------------------------------
    {
        const Grid u = seed_nutrient({NutrientStart::Uniform, 0.75f}, 16, 1);
        expect(u.width == 16 && u.height == 16 &&
                   std::all_of(u.cells.begin(), u.cells.end(), [](float v) { return v == 0.75f; }),
               "uniform nutrient: the level everywhere");
        const Grid g = seed_nutrient({NutrientStart::Gradient, 0.5f}, 9, 1);
        bool rising = true;
        for (std::size_t y = 0; y < 9; ++y) {
            for (std::size_t x = 1; x < 9; ++x) rising = rising && g.at(x, y) > g.at(x - 1, y);
        }
        expect(g.at(0, 3) == 0.0f && g.at(8, 3) == 0.5f && rising, "gradient nutrient: 0 at the left to the level");
        const Grid r1 = seed_nutrient({NutrientStart::Random, 0.8f}, 64, 7);
        const Grid r2 = seed_nutrient({NutrientStart::Random, 0.8f}, 64, 7);
        const Grid r3 = seed_nutrient({NutrientStart::Random, 0.8f}, 64, 8);
        const FieldStats s = field_stats(r1);
        expect(same(r1, r2) && !same(r1, r3), "random nutrient: deterministic per random seed");
        expect(s.min >= 0.0f && s.max < 0.8f && s.mean > 0.36 && s.mean < 0.44, "random nutrient: in [0, level)");
        expect(seed_nutrient({NutrientStart::Uniform, 3.0f}, 2, 0).cells[0] == 1.0f, "the level is clamped to [0, 1]");
    }
    {
        Grid g(2, 2);
        g.cells = {0.5f, -1.0f, 2.0f, 0.5f};
        const FieldStats s = field_stats(g);
        expect(s.min == -1.0f && s.max == 2.0f && s.sum == 2.0 && s.mean == 0.5, "field_stats");
    }

    // --- Simultaneous update: a swap rule exchanges the fields exactly. ------------------------
    const Grid moore = moore_kernel();
    {
        Graph g;
        const int a = g.add_node(NodeKind::Density);
        const int n = g.add_node(NodeKind::Nutrient);
        const int out = g.add_node(NodeKind::Output);
        expect(g.connect(n, 0, out, output_node::NextIn) == ConnectResult::Ok, "Nutrient -> Next state");
        expect(g.connect(a, 0, out, output_node::NutrientIn) == ConnectResult::Ok, "Density -> Next nutrient");
        const GraphFrame f = evaluate(g, 0);
        expect(f.rule_connected && f.nutrient_connected, "evaluate reports both outputs linked");
        const WorldState w{random_grid(13, 9, 1u), random_grid(13, 9, 2u)};
        const WorldState once = step_graph(w, g, moore, GrowthParams{}, 0);
        expect(same(once.density, w.nutrient) && same(once.nutrient, w.density),
               "next A = N and next N = A from the same current state (not one after the other)");
        const WorldState twice = step_graph(once, g, moore, GrowthParams{}, 1);
        expect(same(twice.density, w.density) && same(twice.nutrient, w.nutrient), "two swaps restore both fields");

        const RuleProgram p = compile_rule(g, 0, {}).program;
        expect(p.output != kNoRegister && p.nutrient != kNoRegister && p.output != p.nutrient,
               "the program has a register for each field");
        const std::vector<std::uint32_t> words = pack_rule(p);
        expect(words[1] == p.output && words[3] == p.nutrient, "pack_rule: the nutrient register is header word 3");
        const RuleResult r = run_rule(p, RuleCell{0.25f, 0.75f, 3.0f}, 0, 0, 4, 4);
        expect(r.density == 0.75f && r.nutrient == 0.25f, "run_rule on one cell");
    }
    {
        // Both outputs from one node: compiled once, shared.
        Graph g;
        const int n = g.add_node(NodeKind::Nutrient);
        const int add = g.add_node(NodeKind::Add);
        g.connect(n, 0, add, add_node::A);
        g.set_param(add, add_node::B, 0.125f);
        const int out = g.add_node(NodeKind::Output);
        g.connect(add, 0, out, output_node::NextIn);
        g.connect(add, 0, out, output_node::NutrientIn);
        const RuleProgram p = compile_rule(g, 0, {}).program;
        expect(p.output == p.nutrient && p.ops.size() == 3, "a node feeding both outputs is compiled once");
    }

    // --- Next nutrient unlinked keeps it; alone it is refused. -----------------------------------
    Pattern orbium;
    try {
        orbium = load_pattern_file(std::string(LENIA_PATTERN_DIR) + "/orbium.lenia");
    } catch (const std::exception& ex) {
        std::printf("FAILED: %s\n", ex.what());
        return 1;
    }
    const Grid kernel = normalize_kernel(make_kernel(orbium.kernel), KernelNormalization::Sum);
    Grid start(64, 64, 0.0f);
    stamp_pattern(start, orbium.cells, 32, 32);
    {
        const Graph idle = make_preset(GraphPreset::Idle);
        WorldState w{start, random_grid(64, 64, 3u)};
        Grid single = start;
        for (std::uint64_t gen = 0; gen < 10; ++gen) {
            w = step_graph(w, idle, kernel, orbium.growth, gen);
            single = step_graph(single, idle, kernel, orbium.growth, gen);
        }
        expect(same(w.density, single), "a one-field rule: the density runs exactly as with one field");
        expect(same(w.nutrient, random_grid(64, 64, 3u)), "Next nutrient unlinked: the nutrient is kept bit for bit");

        Graph builtin;
        builtin.add_node(NodeKind::Output);
        const WorldState b = step_graph(WorldState{start, Grid(64, 64, 0.5f)}, builtin, kernel, orbium.growth, 0);
        expect(same(b.density, step_lenia(start, kernel, orbium.growth)) && same(b.nutrient, Grid(64, 64, 0.5f)),
               "the built-in update steps the density and keeps the nutrient");
    }
    {
        Graph g;
        const int a = g.add_node(NodeKind::Density);
        const int out = g.add_node(NodeKind::Output);
        g.connect(a, 0, out, output_node::NutrientIn);
        const RuleCompile c = compile_rule(g, 0, {});
        expect(c.linked && !c.error.empty() && c.program.ops.empty(), "Next nutrient without Next state: an error");
        bool threw = false;
        try {
            step_graph(WorldState{Grid(4, 4), Grid(4, 4)}, g, moore, GrowthParams{}, 0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        expect(threw, "step_graph refuses it");
        bool size_threw = false;
        try {
            step_rule(WorldState{Grid(4, 4), Grid(4, 5)}, moore, RuleProgram{});
        } catch (const std::invalid_argument&) {
            size_threw = true;
        }
        expect(size_threw, "fields of different sizes are refused");
    }

    // --- The nutrient trail demo. ----------------------------------------------------------------
    {
        const Graph trail = make_preset(GraphPreset::NutrientTrail);
        expect(compile_rule(trail, 0, orbium.growth).error.empty(), "the trail rule compiles");
        Graph lenia_rule;
        lenia_rule.add_node(NodeKind::Output);
        wire_lenia_rule(lenia_rule);

        // Full nutrient: the first generation is exactly the Lenia rule's.
        const WorldState full{start, Grid(64, 64, 1.0f)};
        const WorldState t1 = step_graph(full, trail, kernel, orbium.growth, 0);
        const WorldState l1 = step_graph(full, lenia_rule, kernel, orbium.growth, 0);
        expect(same(t1.density, l1.density), "trail at full nutrient: generation 1 is exactly Lenia's");

        // Without uptake the nutrient stays exactly 1 and the density stays Lenia's.
        Graph no_uptake = trail;
        no_uptake.set_param(uptake_node(no_uptake), multiply_node::B, 0.0f);
        WorldState a = full, b = full;
        for (std::uint64_t gen = 0; gen < 20; ++gen) {
            a = step_graph(a, no_uptake, kernel, orbium.growth, gen);
            b = step_graph(b, lenia_rule, kernel, orbium.growth, gen);
        }
        expect(same(a.density, b.density) && same(a.nutrient, full.nutrient),
               "no uptake: nutrient stays 1, the density runs exactly as Lenia for 20 generations");

        // The demo itself: depletion only where the density has been, and Orbium survives.
        WorldState w = full;
        std::vector<char> visited(w.density.cells.size(), 0);
        const double mass0 = field_stats(start).sum;
        bool in_range = true;
        for (std::uint64_t gen = 0; gen < 150; ++gen) {
            for (std::size_t i = 0; i < visited.size(); ++i) visited[i] |= w.density.cells[i] > 0.0f;
            w = step_graph(w, trail, kernel, orbium.growth, gen);
            const FieldStats n = field_stats(w.nutrient);
            in_range = in_range && n.min >= 0.0f && n.max <= 1.0f;
        }
        bool only_visited = true;
        for (std::size_t i = 0; i < visited.size(); ++i) only_visited = only_visited && (visited[i] || w.nutrient.cells[i] == 1.0f);
        const FieldStats n = field_stats(w.nutrient);
        const double mass = field_stats(w.density).sum;
        std::printf("trail after 150 generations: density mass %.1f (start %.1f), nutrient min %.3f mean %.4f\n", mass,
                    mass0, static_cast<double>(n.min), n.mean);
        expect(in_range, "the nutrient stays within [0, 1]");
        expect(n.min < 0.8f && n.mean < 1.0, "the trail consumes over a fifth of the local nutrient");
        expect(only_visited, "uptake depends on the density: untouched cells keep N = 1 exactly");
        expect(mass > 0.25 * mass0 && mass < 4.0 * mass0, "Orbium survives on the nutrient field");

        // Recovery: with the density gone, every depleted cell regrows towards 1.
        WorldState empty{Grid(64, 64, 0.0f), w.nutrient};
        for (std::uint64_t gen = 150; gen < 160; ++gen) empty = step_graph(empty, trail, kernel, orbium.growth, gen);
        bool recovers = true;
        for (std::size_t i = 0; i < w.nutrient.cells.size(); ++i) {
            if (w.nutrient.cells[i] < 0.99f) recovers = recovers && empty.nutrient.cells[i] > w.nutrient.cells[i];
        }
        expect(recovers && field_stats(empty.density).sum == 0.0, "recovery depends on the nutrient: it regrows");

        // Deterministic: the same run twice is bit for bit the same.
        WorldState again = full;
        for (std::uint64_t gen = 0; gen < 150; ++gen) again = step_graph(again, trail, kernel, orbium.growth, gen);
        expect(same(again.density, w.density) && same(again.nutrient, w.nutrient), "the demo replays bit for bit");
    }

    // --- Display: the Nutrient field and the Density over nutrient example. ---------------------
    {
        const Graph d = make_display_preset(DisplayPreset::DensityNutrient);
        const DisplayCompile c = compile_display(d, 0);
        expect(c.linked && c.error.empty(), "the Density over nutrient display compiles");
        const auto full = run_display(c.program, 0.0f, 0.0f, 0.0f, 1.0f);
        const auto empty = run_display(c.program, 0.0f, 0.0f, 0.0f, 0.0f);
        const auto creature = run_display(c.program, 1.0f, 0.0f, 0.0f, 1.0f);
        expect(full != empty, "nutrient shows as soil color");
        expect(creature[0] > 0.9f && creature[1] > 0.9f, "a full density cell is bright over the soil");
        Graph wired = make_display_preset(DisplayPreset::WiredPanel, DisplayField::Nutrient,
                                          {DisplayPalette::Grayscale, 0.0f, 1.0f});
        const auto half = run_display(compile_display(wired, 0).program, 0.0f, 0.0f, 0.0f, 0.5f);
        expect(std::fabs(half[0] - 0.5f) < 0.01f, "the panel's Nutrient field wires to Fields.Nutrient");
    }

    // --- Experiments and snapshots. --------------------------------------------------------------
    {
        Project p = make_nutrient_trail_project(orbium);
        p.nutrient = {NutrientStart::Random, 0.625f};
        p.display[3] = {DisplayPalette::IceFire, 0.125f, 0.875f};
        p.world_size = 48;
        const std::string text = write_project(p);
        const Project q = parse(text);
        expect(q.nutrient == p.nutrient, "[nutrient] roundtrips");
        expect(q.display[3].palette == DisplayPalette::IceFire && q.display[3].lo == 0.125f && q.field == DisplayField::Nutrient,
               "the nutrient display range and field roundtrip");
        expect(q.graph == p.graph && q.display_graph == p.display_graph, "the two-field graphs roundtrip");
        expect(write_project(q) == text, "write(parse(write(p))) == write(p)");
        expect(same(project_nutrient(q), seed_nutrient(p.nutrient, 48, p.random_seed)), "project_nutrient");

        std::printf("Version gates:\n");
        const std::string v3 = replace_line(text, "lenia-project", "lenia-project 3");
        expect(rejected(v3), "a version 3 file with [nutrient] is refused");
        Project plain;
        plain.graph = make_preset(GraphPreset::Idle);
        std::string old = write_project(plain);
        old = replace_line(old, "lenia-project", "lenia-project 3");
        old = replace_line(old, "nutrient ", ""); // the display range line
        const auto cut = old.find("[nutrient]");
        old.erase(cut, old.find("[", cut + 1) - cut);
        try {
            const Project r = parse(old);
            expect(r.nutrient == NutrientSeed{} && r.graph == plain.graph, "a version 3 file reads with a uniform nutrient");
        } catch (const std::exception& ex) {
            std::printf("FAILED: version 3 file: %s\n", ex.what());
            ++failures;
        }
        expect(rejected(old + "node 999 nutrient 0 0\n"), "a Nutrient node in a version 3 file");
        expect(rejected(replace_line(text, "level", "level 1.5")), "a nutrient level above 1");
        expect(rejected(replace_line(text, "start", "start spots")), "an unknown nutrient start");

        // Snapshot: both fields bit for bit; the continuation is exact.
        Grid density(48, 48, 0.0f);
        stamp_pattern(density, orbium.cells, 24, 24);
        WorldState w{density, project_nutrient(p)};
        for (std::uint64_t gen = 0; gen < 5; ++gen) w = step_graph(w, p.graph, kernel, p.growth, gen);
        Snapshot s{p, 5, w.density, w.nutrient};
        const std::string bytes = write_snapshot(s);
        const Snapshot r = parse_snapshot(bytes);
        expect(r.generation == 5 && same(r.density, w.density) && same(r.nutrient, w.nutrient),
               "a snapshot stores both fields bit for bit");
        WorldState resumed{r.density, r.nutrient}, original = w;
        for (std::uint64_t gen = 5; gen < 10; ++gen) {
            resumed = step_graph(resumed, r.project.graph, kernel, r.project.growth, gen);
            original = step_graph(original, p.graph, kernel, p.growth, gen);
        }
        expect(same(resumed.density, original.density) && same(resumed.nutrient, original.nutrient),
               "a loaded two-field snapshot continues exactly");

        // A version 1 snapshot (no nutrient block) of a single-field experiment still opens.
        Project single;
        single.world_size = 8;
        single.graph = make_preset(GraphPreset::Idle);
        const std::string v2 = write_snapshot(Snapshot{single, 3, Grid(8, 8, 0.25f), {}});
        std::string v1 = v2;
        v1.replace(0, v1.find('\n'), "lenia-snapshot 1");
        const std::string experiment = write_project(single);
        std::string old_experiment = replace_line(experiment, "lenia-project", "lenia-project 3");
        old_experiment = replace_line(old_experiment, "nutrient ", "");
        const auto at = old_experiment.find("[nutrient]");
        old_experiment.erase(at, old_experiment.find("[", at + 1) - at);
        const auto e_line = v1.find("\nexperiment ") + 1;
        const auto e_data = v1.find('\n', e_line) + 1;
        const auto n_line = v1.find("nutrient 8 8 f32le\n", e_data + experiment.size());
        std::string body = v1.substr(0, e_line) + "experiment " + std::to_string(old_experiment.size()) + "\n" +
                           old_experiment + v1.substr(e_data + experiment.size(), n_line - (e_data + experiment.size()));
        char hex[17];
        std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(snapshot_checksum(body)));
        body += std::string("checksum fnv1a64 ") + hex + "\n";
        try {
            const Snapshot o = parse_snapshot(body);
            expect(o.generation == 3 && same(o.density, Grid(8, 8, 0.25f)) && same(o.nutrient, Grid(8, 8, 1.0f)),
                   "a version 1 snapshot opens with its experiment's starting nutrient");
        } catch (const std::exception& ex) {
            std::printf("FAILED: version 1 snapshot: %s\n", ex.what());
            ++failures;
        }
        Snapshot bad{p, 5, w.density, Grid(48, 48, 0.0f)};
        bad.nutrient.cells[7] = std::nanf("");
        bool threw = false;
        try {
            write_snapshot(bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        expect(threw, "a NaN nutrient cell is not written");
    }

    if (failures == 0) std::printf("All state field tests passed.\n");
    return failures == 0 ? 0 : 1;
}
