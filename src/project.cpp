#include "lenia/project.hpp"

#include "lenia/rule.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

namespace lenia {

namespace {

// Sanity limits, far above what the workbench uses, so a corrupt file cannot request huge
// allocations. The workbench applies its own, tighter limits on top (main_window.cpp).
constexpr int kMaxRadius = 1024;
constexpr std::size_t kMaxRings = 256;
constexpr int kMaxGridSide = 4096;
constexpr int kMaxWorldSize = 8192;
constexpr std::size_t kMaxNodes = 4096;
// Paint stores raw weight minus the shape's weight, both in [-1, 1].
constexpr float kMaxPaint = 2.0f;

// Tokens, indexed by the enum value.
const std::vector<std::string> kCoreNames = {"exponential", "polynomial", "step", "gaussian"};
const std::vector<std::string> kNormalizationNames = {"none", "sum", "positive", "abs_sum"};
const std::vector<std::string> kGrowthNames = {"gaussian", "polynomial"};
const std::vector<std::string> kSeedNames = {"random", "pattern", "impulse", "empty"};
const std::vector<std::string> kFieldNames = {"density", "neighborhood", "growth"};
const std::vector<std::string> kPaletteNames = {"grayscale", "viridis", "inferno", "redblue", "icefire"};
const std::vector<std::string> kKindNames = {"time",     "oscillator", "constant", "kernel",  "environment",
                                             "output",   "density",    "neighborhood", "growth", "add",
                                             "multiply", "clamp",      "compare",  "select",  "life",
                                             "fields",   "ramp",       "color",    "mix",     "view"};
// Version 1 knew only the first six kinds, and fewer ports on the Environment and Output nodes.
constexpr std::size_t kVersion1Kinds = 6;
// Display graphs ([display_graph]) arrived with version 3.
constexpr int kDisplayGraphVersion = 3;

template <class E> const std::string& name_of(const std::vector<std::string>& names, E value) {
    return names.at(static_cast<std::size_t>(value));
}

// ---------------------------------------------------------------------------------------------
// Writing

std::string num(float v) {
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, v); // shortest form that reads back exactly
    return std::string(buf, r.ptr);
}

// Text values are the rest of their line, trimmed.
std::string text(const std::string& s) {
    std::string out = s;
    std::replace_if(out.begin(), out.end(), [](char c) { return c == '\n' || c == '\r'; }, ' ');
    const auto first = out.find_first_not_of(" \t");
    if (first == std::string::npos) return {};
    return out.substr(first, out.find_last_not_of(" \t") - first + 1);
}

void write_grid(std::ostringstream& out, const std::string& key, const Grid& g) {
    out << key << ' ' << g.width << ' ' << g.height << '\n';
    for (std::size_t y = 0; y < g.height; ++y) {
        for (std::size_t x = 0; x < g.width; ++x) out << (x ? " " : "") << num(g.at(x, y));
        out << '\n';
    }
}

void write_graph(std::ostringstream& out, const Graph& g) {
    for (const Node& n : g.nodes()) {
        out << "node " << n.id << ' ' << name_of(kKindNames, n.kind) << ' ' << num(n.x) << ' ' << num(n.y);
        for (float v : n.params) out << ' ' << num(v);
        out << '\n';
    }
    for (const Link& l : g.links()) {
        out << "link " << l.from_node << ' ' << l.from_port << ' ' << l.to_node << ' ' << l.to_port << '\n';
    }
}

// ---------------------------------------------------------------------------------------------
// Parsing

struct Line {
    int number = 0;
    std::vector<std::string> tokens; // whitespace-separated, at least one
    std::string rest;                // everything after the first token, trimmed
};

[[noreturn]] void fail(int line, const std::string& message) {
    throw std::invalid_argument("project, line " + std::to_string(line) + ": " + message);
}

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument("project: " + message); }

std::vector<Line> read_lines(std::istream& in) {
    std::vector<Line> lines;
    std::string s;
    int number = 0;
    while (std::getline(in, s)) {
        ++number;
        const auto first = s.find_first_not_of(" \t\r");
        if (first == std::string::npos || s[first] == '#') continue; // blank or comment
        Line l;
        l.number = number;
        std::istringstream fields(s);
        std::string token;
        while (fields >> token) l.tokens.push_back(token);
        const auto key_end = s.find_first_of(" \t\r", first);
        if (key_end != std::string::npos) l.rest = text(s.substr(key_end));
        lines.push_back(std::move(l));
    }
    return lines;
}

bool is_section(const Line& l) { return l.tokens[0].size() >= 2 && l.tokens[0].front() == '['; }

float parse_float(const Line& l, const std::string& token) {
    float v = 0.0f;
    const auto [end, ec] = std::from_chars(token.data(), token.data() + token.size(), v);
    if (ec != std::errc() || end != token.data() + token.size() || !std::isfinite(v)) {
        fail(l.number, "'" + token + "' is not a finite number");
    }
    return v;
}

int parse_int(const Line& l, const std::string& token, int lo, int hi) {
    int v = 0;
    const auto [end, ec] = std::from_chars(token.data(), token.data() + token.size(), v);
    if (ec != std::errc() || end != token.data() + token.size()) fail(l.number, "'" + token + "' is not a whole number");
    if (v < lo || v > hi) {
        fail(l.number, "'" + token + "' is out of range [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    }
    return v;
}

int parse_choice(const Line& l, const std::string& token, const std::vector<std::string>& names) {
    const auto it = std::find(names.begin(), names.end(), token);
    if (it == names.end()) {
        std::string list;
        for (const std::string& n : names) list += (list.empty() ? "" : ", ") + n;
        fail(l.number, "unknown " + l.tokens[0] + " '" + token + "' (expected one of: " + list + ")");
    }
    return static_cast<int>(it - names.begin());
}

void expect_values(const Line& l, std::size_t n) {
    if (l.tokens.size() != n + 1) {
        fail(l.number, l.tokens[0] + " takes " + std::to_string(n) + (n == 1 ? " value" : " values"));
    }
}

class Parser {
public:
    explicit Parser(std::vector<Line> lines) : lines_(std::move(lines)) {}

    Project parse() {
        read_header();
        Project p;
        std::set<std::string> seen;
        while (pos_ < lines_.size()) {
            const Line& h = lines_[pos_++];
            if (!is_section(h) || h.tokens.size() != 1 || h.tokens[0].back() != ']') {
                fail(h.number, "expected a section header such as [kernel], got '" + h.tokens[0] + "'");
            }
            const std::string name = h.tokens[0].substr(1, h.tokens[0].size() - 2);
            if (!seen.insert(name).second) fail(h.number, "section [" + name + "] appears twice");
            if (name == "kernel") parse_kernel(h, p);
            else if (name == "growth") parse_growth(h, p);
            else if (name == "world") parse_world(h, p);
            else if (name == "pattern") parse_pattern_section(h, p);
            else if (name == "display") parse_display(h, p);
            else if (name == "graph") parse_graph(h, p.graph, "graph");
            else if (name == "display_graph" && version_ >= kDisplayGraphVersion) {
                parse_graph(h, p.display_graph, "display_graph");
            } else if (name == "display_graph") {
                fail(h.number, "[display_graph] needs format version " + std::to_string(kDisplayGraphVersion));
            } else {
                fail(h.number, "unknown section [" + name + "]");
            }
        }
        for (const char* required : {"kernel", "growth", "world", "display", "graph"}) {
            if (!seen.count(required)) fail(std::string("missing section [") + required + "]");
        }
        // Older files have no display graph: the default (a lone View node) shows the panel display.
        if (version_ >= kDisplayGraphVersion && !seen.count("display_graph")) fail("missing section [display_graph]");
        if (p.paint.width != 0 && p.paint.width != static_cast<std::size_t>(2 * p.kernel.radius + 1)) {
            fail("kernel paint is " + std::to_string(p.paint.width) + " cells wide; radius " +
                 std::to_string(p.kernel.radius) + " needs " + std::to_string(2 * p.kernel.radius + 1));
        }
        if (p.pattern) {
            p.pattern->kernel = p.kernel;
            p.pattern->growth = p.growth;
        }
        return p;
    }

private:
    // Key lines of one section, each key at most once and all of them known. Keys in `grids`
    // ("key W H", or "key none") are followed by H rows of W numbers, returned in `grid_values`.
    struct Section {
        int line = 0;
        std::map<std::string, Line> keys;
        std::map<std::string, Grid> grids;

        const Line& get(const std::string& key, const std::string& section) const {
            const auto it = keys.find(key);
            if (it == keys.end()) fail(line, "[" + section + "] is missing '" + key + "'");
            return it->second;
        }
    };

    void read_header() {
        if (lines_.empty()) fail("the file is empty");
        const Line& h = lines_[pos_++];
        if (h.tokens[0] != "lenia-project" || h.tokens.size() != 2) {
            fail(h.number, "not a Lenia project (the first line must be 'lenia-project <version>')");
        }
        int version = 0;
        const std::string& v = h.tokens[1];
        const auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(), version);
        if (ec != std::errc() || end != v.data() + v.size() || version < 1) {
            fail(h.number, "bad format version '" + v + "'");
        }
        if (version > kProjectVersion) {
            fail(h.number, "format version " + v + " is newer than this workbench reads (up to " +
                               std::to_string(kProjectVersion) + ")");
        }
        version_ = version;
    }

    Section read_section(const Line& header, const std::string& name, const std::set<std::string>& allowed,
                         const std::set<std::string>& grids = {}) {
        Section s;
        s.line = header.number;
        while (pos_ < lines_.size() && !is_section(lines_[pos_])) {
            const Line& l = lines_[pos_++];
            const std::string& key = l.tokens[0];
            if (!allowed.count(key)) fail(l.number, "unknown key '" + key + "' in [" + name + "]");
            if (!s.keys.emplace(key, l).second) fail(l.number, "'" + key + "' appears twice in [" + name + "]");
            if (grids.count(key)) s.grids[key] = read_grid(l);
        }
        return s;
    }

    Grid read_grid(const Line& l) {
        if (l.tokens.size() == 2 && l.tokens[1] == "none") return {};
        expect_values(l, 2);
        const int w = parse_int(l, l.tokens[1], 1, kMaxGridSide);
        const int h = parse_int(l, l.tokens[2], 1, kMaxGridSide);
        Grid g;
        g.width = static_cast<std::size_t>(w);
        g.height = static_cast<std::size_t>(h);
        for (int y = 0; y < h; ++y) {
            if (pos_ >= lines_.size() || is_section(lines_[pos_])) {
                fail(l.number, l.tokens[0] + " has " + std::to_string(y) + " rows, expected " + std::to_string(h));
            }
            const Line& row = lines_[pos_++];
            if (row.tokens.size() != g.width) {
                fail(row.number, l.tokens[0] + " row has " + std::to_string(row.tokens.size()) + " values, expected " +
                                     std::to_string(w));
            }
            for (const std::string& t : row.tokens) g.cells.push_back(parse_float(row, t));
        }
        return g;
    }

    void parse_kernel(const Line& header, Project& p) {
        const Section s = read_section(header, "kernel", {"radius", "core", "rings", "normalization", "paint"}, {"paint"});
        const Line& radius = s.get("radius", "kernel");
        expect_values(radius, 1);
        p.kernel.radius = parse_int(radius, radius.tokens[1], 1, kMaxRadius);

        const Line& core = s.get("core", "kernel");
        expect_values(core, 1);
        p.kernel.core = static_cast<KernelCore>(parse_choice(core, core.tokens[1], kCoreNames));

        const Line& rings = s.get("rings", "kernel");
        if (rings.tokens.size() < 2 || rings.tokens.size() > kMaxRings + 1) {
            fail(rings.number, "rings takes 1 to " + std::to_string(kMaxRings) + " weights");
        }
        p.kernel.rings.clear();
        for (std::size_t i = 1; i < rings.tokens.size(); ++i) p.kernel.rings.push_back(parse_float(rings, rings.tokens[i]));

        const Line& norm = s.get("normalization", "kernel");
        expect_values(norm, 1);
        p.normalization = static_cast<KernelNormalization>(parse_choice(norm, norm.tokens[1], kNormalizationNames));

        const Line& paint = s.get("paint", "kernel");
        p.paint = s.grids.at("paint");
        if (p.paint.width != p.paint.height) fail(paint.number, "paint must be square");
        for (float v : p.paint.cells) {
            if (std::fabs(v) > kMaxPaint) fail(paint.number, "paint values must be within [-2, 2]");
        }
    }

    void parse_growth(const Line& header, Project& p) {
        const Section s = read_section(header, "growth", {"function", "mu", "sigma", "dt"});
        const Line& function = s.get("function", "growth");
        expect_values(function, 1);
        p.growth.function = static_cast<GrowthFunction>(parse_choice(function, function.tokens[1], kGrowthNames));
        const auto value = [&](const char* key) {
            const Line& l = s.get(key, "growth");
            expect_values(l, 1);
            return std::pair<const Line&, float>(l, parse_float(l, l.tokens[1]));
        };
        p.growth.mu = value("mu").second;
        const auto sigma = value("sigma");
        if (!(sigma.second > 0.0f)) fail(sigma.first.number, "sigma must be positive");
        p.growth.sigma = sigma.second;
        const auto dt = value("dt");
        if (!(dt.second > 0.0f)) fail(dt.first.number, "dt must be positive");
        p.growth.dt = dt.second;
    }

    void parse_world(const Line& header, Project& p) {
        const Section s = read_section(header, "world", {"size", "seed", "random_seed"});
        const Line& size = s.get("size", "world");
        expect_values(size, 1);
        p.world_size = parse_int(size, size.tokens[1], 1, kMaxWorldSize);
        const Line& seed = s.get("seed", "world");
        expect_values(seed, 1);
        p.seed = static_cast<SeedKind>(parse_choice(seed, seed.tokens[1], kSeedNames));
        const Line& random = s.get("random_seed", "world");
        expect_values(random, 1);
        p.random_seed = parse_int(random, random.tokens[1], 0, 2147483647);
    }

    void parse_pattern_section(const Line& header, Project& p) {
        const Section s = read_section(header, "pattern", {"name", "source", "license", "cells"}, {"cells"});
        Pattern pattern;
        if (const auto it = s.keys.find("name"); it != s.keys.end()) pattern.name = it->second.rest;
        if (const auto it = s.keys.find("source"); it != s.keys.end()) pattern.source = it->second.rest;
        if (const auto it = s.keys.find("license"); it != s.keys.end()) pattern.license = it->second.rest;
        const Line& cells = s.get("cells", "pattern");
        pattern.cells = s.grids.at("cells");
        if (pattern.cells.width == 0) fail(cells.number, "cells needs a width and height");
        p.pattern = std::move(pattern);
    }

    void parse_display(const Line& header, Project& p) {
        std::set<std::string> keys(kFieldNames.begin(), kFieldNames.end());
        keys.insert("field");
        const Section s = read_section(header, "display", keys);
        const Line& field = s.get("field", "display");
        expect_values(field, 1);
        p.field = static_cast<DisplayField>(parse_choice(field, field.tokens[1], kFieldNames));
        for (std::size_t i = 0; i < kFieldNames.size(); ++i) {
            const Line& l = s.get(kFieldNames[i], "display");
            expect_values(l, 3);
            DisplayRange& r = p.display[i];
            r.palette = static_cast<DisplayPalette>(parse_choice(l, l.tokens[1], kPaletteNames));
            r.lo = parse_float(l, l.tokens[2]);
            r.hi = parse_float(l, l.tokens[3]);
        }
    }

    // [graph] or [display_graph], into `graph` (whose domain decides which node kinds belong).
    void parse_graph(const Line& header, Graph& graph, const std::string& section) {
        const bool display = graph.domain() == GraphDomain::Display;
        std::vector<Node> nodes;
        std::vector<Link> links;
        while (pos_ < lines_.size() && !is_section(lines_[pos_])) {
            const Line& l = lines_[pos_++];
            if (l.tokens[0] == "node") {
                if (l.tokens.size() < 5) fail(l.number, "node takes an id, a kind, x, y and the parameters");
                if (nodes.size() >= kMaxNodes) fail(l.number, "too many nodes");
                Node n;
                n.id = parse_int(l, l.tokens[1], 1, 1 << 30);
                const int kind = parse_choice(l, l.tokens[2], kKindNames);
                if (!in_domain(static_cast<NodeKind>(kind), graph.domain())) {
                    fail(l.number, "node kind '" + l.tokens[2] + "' does not belong in [" + section + "]" +
                                       (display ? "" : " (display nodes go in [display_graph])"));
                }
                if (version_ < 2 && static_cast<std::size_t>(kind) >= kVersion1Kinds) {
                    fail(l.number, "node kind '" + l.tokens[2] + "' needs format version 2");
                }
                n.kind = static_cast<NodeKind>(kind);
                n.x = parse_float(l, l.tokens[3]);
                n.y = parse_float(l, l.tokens[4]);
                for (std::size_t i = 5; i < l.tokens.size(); ++i) n.params.push_back(parse_float(l, l.tokens[i]));
                nodes.push_back(std::move(n));
            } else if (l.tokens[0] == "link") {
                expect_values(l, 4);
                if (links.size() >= kMaxNodes * 4) fail(l.number, "too many links");
                Link k{};
                k.from_node = parse_int(l, l.tokens[1], 1, 1 << 30);
                k.from_port = parse_int(l, l.tokens[2], 0, 64);
                k.to_node = parse_int(l, l.tokens[3], 1, 1 << 30);
                k.to_port = parse_int(l, l.tokens[4], 0, 64);
                links.push_back(k);
            } else {
                fail(l.number, "unknown key '" + l.tokens[0] + "' in [" + section + "] (expected node or link)");
            }
        }
        const std::string error = graph.assign(nodes, links);
        if (!error.empty()) fail(header.number, "[" + section + "]: " + error);
        if (version_ < 2 && !display) {
            for (const Link& k : graph.links()) {
                const bool field = graph.node(k.from_node)->kind == NodeKind::Environment &&
                                   k.from_port == environment_node::FieldOut;
                const bool next = k.to_node == graph.output_node() && k.to_port == output_node::NextIn;
                if (field || next) {
                    fail(header.number, "[graph]: link " + std::to_string(k.from_node) + ":" +
                                            std::to_string(k.from_port) + " -> " + std::to_string(k.to_node) + ":" +
                                            std::to_string(k.to_port) + " needs format version 2");
                }
            }
        }
    }

    std::vector<Line> lines_;
    std::size_t pos_ = 0;
    int version_ = kProjectVersion;
};

} // namespace

std::string write_project(const Project& p) {
    std::ostringstream out;
    out << "lenia-project " << kProjectVersion << "\n"
        << "# Lenia workbench experiment (docs/projects.md). Opening it restarts at generation 0.\n";

    out << "\n[kernel]\n"
        << "radius " << p.kernel.radius << '\n'
        << "core " << name_of(kCoreNames, p.kernel.core) << '\n'
        << "rings";
    for (float w : p.kernel.rings) out << ' ' << num(w);
    out << '\n' << "normalization " << name_of(kNormalizationNames, p.normalization) << '\n';
    if (p.paint.cells.empty()) out << "paint none\n";
    else write_grid(out, "paint", p.paint);

    out << "\n[growth]\n"
        << "function " << name_of(kGrowthNames, p.growth.function) << '\n'
        << "mu " << num(p.growth.mu) << '\n'
        << "sigma " << num(p.growth.sigma) << '\n'
        << "dt " << num(p.growth.dt) << '\n';

    out << "\n[world]\n"
        << "size " << p.world_size << '\n'
        << "seed " << name_of(kSeedNames, p.seed) << '\n'
        << "random_seed " << p.random_seed << '\n';

    if (p.pattern) {
        out << "\n[pattern]\n";
        if (!text(p.pattern->name).empty()) out << "name " << text(p.pattern->name) << '\n';
        if (!text(p.pattern->source).empty()) out << "source " << text(p.pattern->source) << '\n';
        if (!text(p.pattern->license).empty()) out << "license " << text(p.pattern->license) << '\n';
        write_grid(out, "cells", p.pattern->cells);
    }

    out << "\n[display]\n"
        << "field " << name_of(kFieldNames, p.field) << '\n';
    for (std::size_t i = 0; i < kFieldNames.size(); ++i) {
        const DisplayRange& r = p.display[i];
        out << kFieldNames[i] << ' ' << name_of(kPaletteNames, r.palette) << ' ' << num(r.lo) << ' ' << num(r.hi)
            << '\n';
    }

    out << "\n[display_graph]\n"
        << "# colors the view; a View node with its input unlinked shows the panel display above\n";
    write_graph(out, p.display_graph);

    out << "\n[graph]\n"
        << "# node <id> <kind> <x> <y> <parameters in order>; link <from node> <output> <to node> <input>\n";
    write_graph(out, p.graph);
    return out.str();
}

Project parse_project(std::istream& in) { return Parser(read_lines(in)).parse(); }

Project make_life_project(GraphPreset preset) {
    Project p;
    // Radius 1: make_kernel's ring shape has no nonzero cell (the middle is at t = 0, the
    // neighbors at r >= 1), so the paint alone is the kernel, and paint = raw weight.
    p.kernel.radius = 1;
    p.kernel.core = KernelCore::Step;
    p.kernel.rings = {1.0f};
    p.normalization = KernelNormalization::None;
    p.paint = moore_kernel();
    p.world_size = 128;
    p.seed = SeedKind::Pattern;
    p.pattern = glider_pattern();
    p.field = DisplayField::Density;
    p.display[0] = {DisplayPalette::Grayscale, 0.0f, 1.0f};
    p.display[1] = {DisplayPalette::Inferno, 0.0f, 8.0f};
    p.display[2] = {DisplayPalette::RedBlue, -1.0f, 1.0f};
    p.graph = make_preset(is_life_preset(preset) ? preset : GraphPreset::GameOfLife);
    return p;
}

} // namespace lenia
