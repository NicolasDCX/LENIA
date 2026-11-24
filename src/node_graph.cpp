#include "lenia/node_graph.hpp"

#include "lenia/rule.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace lenia {

namespace {

constexpr double kTwoPi = 6.283185307179586;
constexpr double kDegree = kTwoPi / 360.0;
// Links cannot form cycles (Graph::connect), so this only bounds pathological chains.
constexpr int kMaxDepth = 256;

std::vector<NodeSpec> build_specs() {
    using T = PortType;
    std::vector<NodeSpec> s;
    s.push_back({NodeKind::Time,
                 "Time",
                 "Outputs Start + Speed × generation. Restarts with the world (Reset, Orbium, load).",
                 {},
                 {{"Generation", T::Value}},
                 {{"Speed", -100.0f, 100.0f, 1.0f, 0.05f, 3, {}}, {"Start", -1e6f, 1e6f, 0.0f, 1.0f, 1, {}}}});
    s.push_back({NodeKind::Oscillator,
                 "Oscillator",
                 "Offset + Amplitude × wave(Time / Period + Phase). Time defaults to the generation.",
                 {{"Time", T::Value, -1},
                  {"Amplitude", T::Value, oscillator_node::Amplitude},
                  {"Offset", T::Value, oscillator_node::Offset}},
                 {{"Value", T::Value}},
                 {{"Wave", 0.0f, 3.0f, 0.0f, 1.0f, 0, {"Sine", "Triangle", "Square", "Saw"}},
                  {"Period (gens)", 1.0f, 1e6f, 240.0f, 10.0f, 1, {}},
                  {"Phase", 0.0f, 1.0f, 0.0f, 0.05f, 3, {}},
                  {"Amplitude", -100.0f, 100.0f, 0.1f, 0.01f, 3, {}},
                  {"Offset", -100.0f, 100.0f, 0.0f, 0.01f, 3, {}}}});
    s.push_back({NodeKind::Constant,
                 "Constant",
                 "A fixed value.",
                 {},
                 {{"Value", T::Value}},
                 {{"Value", -1e6f, 1e6f, 0.1f, 0.01f, 3, {}}}});
    s.push_back({NodeKind::Kernel,
                 "Kernel",
                 "Warps the Kernel panel's ring shape each generation. Radius Δ scales the radius by "
                 "(1 + Δ), Peak shift moves each ring's peak (+ outward), Sharpness narrows (+) or "
                 "widens (−) the rings. Linked inputs add to the parameters; all 0 = the static kernel "
                 "(including paint).",
                 {{"Radius Δ", T::Value, kernel_node::RadiusDelta},
                  {"Peak shift", T::Value, kernel_node::PeakShift},
                  {"Sharpness", T::Value, kernel_node::Sharpness}},
                 {{"Kernel", T::Kernel}},
                 {{"Radius Δ", -0.9f, 3.0f, 0.0f, 0.01f, 3, {}},
                  {"Peak shift", -3.0f, 3.0f, 0.0f, 0.05f, 3, {}},
                  {"Sharpness", -3.0f, 3.0f, 0.0f, 0.05f, 3, {}}}});
    s.push_back({NodeKind::Environment,
                 "Environment",
                 "Spatial growth influence E(x, y) in [−1, 1]. Its Field output is Strength × E per cell, "
                 "for a rule to add to its growth; linked into Simulation instead, the built-in update "
                 "uses G(U) + Strength × E. Offset moves the pattern along Angle (cells); link Time to "
                 "Offset to make it drift.",
                 {{"Strength", T::Value, environment_node::Strength},
                  {"Offset", T::Value, environment_node::Offset},
                  {"Angle", T::Value, environment_node::Angle}},
                 {{"Environment", T::Environment}, {"Field", T::Cell}},
                 {{"Pattern", 0.0f, 4.0f, 3.0f, 1.0f, 0, {"Uniform", "Spot", "Gradient", "Stripes", "Ripples"}},
                  {"Strength", -1.0f, 1.0f, 0.03f, 0.005f, 4, {}},
                  {"Scale (cells)", 1.0f, 4096.0f, 64.0f, 4.0f, 1, {}},
                  {"Angle (°)", -3600.0f, 3600.0f, 0.0f, 5.0f, 1, {}},
                  {"Offset (cells)", -1e6f, 1e6f, 0.0f, 1.0f, 2, {}}}});
    s.push_back({NodeKind::Output,
                 "Simulation",
                 "What the Vulkan simulation uses each generation. Unlinked Kernel = the Kernel panel's "
                 "static kernel. Next state = the per-cell rule's result; unlinked, the built-in Lenia "
                 "update clamp(A + dt × (G(U) + environment), 0, 1) runs with the Simulation panel's growth "
                 "and the Environment linked here (unlinked = none). A linked rule ignores the Environment "
                 "input: wire the Environment's Field into the rule instead. Next nutrient = the nutrient "
                 "field's next value, computed from the same current state as Next state (both update "
                 "together); unlinked, the nutrient stays as it is. It needs Next state linked too.",
                 {{"Kernel", T::Kernel},
                  {"Environment", T::Environment},
                  {"Next state", T::Cell},
                  {"Next nutrient", T::Cell}},
                 {},
                 {}});

    // Cell rule nodes.
    s.push_back({NodeKind::Density,
                 "Density A",
                 "The cell's current state A, as shown in the Density view: in [0, 1] for Lenia, 0 or 1 "
                 "for the Game of Life. The Field output is the whole density, for a Convolve node.",
                 {},
                 {{"A", T::Cell}, {"Field", T::Field}},
                 {}});
    s.push_back({NodeKind::Neighborhood,
                 "Neighborhood U",
                 "U = K ∗ A: the cell's neighbors weighted by the active kernel (Simulation's Kernel input, "
                 "or the Kernel panel's). With raw weights (normalization None) it is a weighted count; "
                 "the Game of Life's Moore kernel counts the 8 live neighbors exactly.",
                 {},
                 {{"U", T::Cell}},
                 {}});
    s.push_back({NodeKind::Growth,
                 "Growth G(U)",
                 "Lenia growth mapping, 2·exp(−z²/2) − 1 or 2·max(0, 1 − z²/9)⁴ − 1 with z = (U − μ) / σ. "
                 "Source = Simulation panel follows the panel's function, μ, σ and dt (Restore Orbium sets "
                 "them); This node uses the parameters below. Unlinked U = the neighborhood; linked μ / σ "
                 "replace the parameter per cell. The dt output is the step size to scale G by.",
                 {{"U", T::Cell, -1}, {"μ", T::Cell, growth_node::Mu}, {"σ", T::Cell, growth_node::Sigma}},
                 {{"G", T::Cell}, {"dt", T::Value}},
                 {{"Source", 0.0f, 1.0f, 0.0f, 1.0f, 0, {"Simulation panel", "This node"}},
                  {"Function", 0.0f, 1.0f, 0.0f, 1.0f, 0, {"Gaussian", "Polynomial"}},
                  {"μ", -10.0f, 10.0f, 0.15f, 0.001f, 4, {}},
                  {"σ", 0.0001f, 10.0f, 0.015f, 0.001f, 4, {}},
                  {"dt", 0.0001f, 10.0f, 0.1f, 0.01f, 4, {}}}});
    s.push_back({NodeKind::Add,
                 "Add",
                 "A + B per cell. Unlinked inputs use the parameters; a link replaces its parameter.",
                 {{"A", T::Cell, add_node::A}, {"B", T::Cell, add_node::B}},
                 {{"A + B", T::Cell}},
                 {{"A", -1e6f, 1e6f, 0.0f, 0.01f, 4, {}}, {"B", -1e6f, 1e6f, 0.0f, 0.01f, 4, {}}}});
    s.push_back({NodeKind::Multiply,
                 "Multiply",
                 "A × B per cell. Unlinked inputs use the parameters; a link replaces its parameter.",
                 {{"A", T::Cell, multiply_node::A}, {"B", T::Cell, multiply_node::B}},
                 {{"A × B", T::Cell}},
                 {{"A", -1e6f, 1e6f, 1.0f, 0.01f, 4, {}}, {"B", -1e6f, 1e6f, 1.0f, 0.01f, 4, {}}}});
    s.push_back({NodeKind::Clamp,
                 "Clamp",
                 "min(max(X, Low), High) per cell; Lenia keeps the state in [0, 1] this way.",
                 {{"X", T::Cell, clamp_node::X}, {"Low", T::Cell, clamp_node::Low}, {"High", T::Cell, clamp_node::High}},
                 {{"Clamped", T::Cell}},
                 {{"X", -1e6f, 1e6f, 0.0f, 0.01f, 4, {}},
                  {"Low", -1e6f, 1e6f, 0.0f, 0.01f, 4, {}},
                  {"High", -1e6f, 1e6f, 1.0f, 0.01f, 4, {}}}});
    s.push_back({NodeKind::Compare,
                 "Compare",
                 "1 where the test holds, else 0. Exact float comparison: with the raw Moore kernel the "
                 "neighbor count is a whole number, so U = 3 is exact.",
                 {{"A", T::Cell, compare_node::A}, {"B", T::Cell, compare_node::B}},
                 {{"1 / 0", T::Cell}},
                 {{"Test", 0.0f, 5.0f, 1.0f, 1.0f, 0, {"A > B", "A ≥ B", "A < B", "A ≤ B", "A = B", "A ≠ B"}},
                  {"A", -1e6f, 1e6f, 0.0f, 0.01f, 4, {}},
                  {"B", -1e6f, 1e6f, 0.5f, 0.01f, 4, {}}}});
    s.push_back({NodeKind::Select,
                 "Select",
                 "If true where Condition ≥ 0.5, else If false (per cell).",
                 {{"Condition", T::Cell, select_node::Condition},
                  {"If true", T::Cell, select_node::IfTrue},
                  {"If false", T::Cell, select_node::IfFalse}},
                 {{"Value", T::Cell}},
                 {{"Condition", -1e6f, 1e6f, 0.0f, 0.01f, 4, {}},
                  {"If true", -1e6f, 1e6f, 1.0f, 0.01f, 4, {}},
                  {"If false", -1e6f, 1e6f, 0.0f, 0.01f, 4, {}}}});
    s.push_back({NodeKind::Life,
                 "Life transition",
                 "Exact life-like rule. A cell is alive when State ≥ 0.5; with n = round(Neighbors), a dead "
                 "cell becomes 1 if bit n of Birth is set, a live one stays 1 if bit n of Survive is set, "
                 "otherwise 0. Masks add 2ⁿ per count n: Conway's B3/S23 is Birth 8, Survive 4 + 8 = 12. "
                 "Unlinked State = the density, Neighbors = the neighborhood.",
                 {{"State", T::Cell, -1}, {"Neighbors", T::Cell, -1}},
                 {{"Next", T::Cell}},
                 {{"Birth mask", 0.0f, 511.0f, 8.0f, 1.0f, 0, {}}, {"Survive mask", 0.0f, 511.0f, 12.0f, 1.0f, 0, {}}}});

    // Display nodes.
    s.push_back({NodeKind::Fields,
                 "Fields",
                 "The simulation fields at each cell: density A, neighborhood U, growth and nutrient N, exactly "
                 "what the panel's Field choice shows (Growth: the rule's first Growth node, or next − current).",
                 {},
                 {{"Density A", T::Cell}, {"Neighborhood U", T::Cell}, {"Growth", T::Cell}, {"Nutrient N", T::Cell}},
                 {}});
    {
        NodeSpec ramp{NodeKind::ColorRamp,
                      "Color ramp",
                      "Maps Value to a color: t = (Value − Low) / (High − Low), clamped to [0, 1], picks a color "
                      "between the stops (linear in sRGB). Unlinked Value = the density. Edit the stops in the "
                      "inspector: drag to move, double-click to add or recolor, right-click to remove.",
                      {{"Value", T::Cell, -1}},
                      {{"Color", T::Color}},
                      {{"Low", -100.0f, 100.0f, 0.0f, 0.01f, 4, {}},
                       {"High", -100.0f, 100.0f, 1.0f, 0.01f, 4, {}},
                       {"Stops", 2.0f, static_cast<float>(ramp_node::kMaxStops), 2.0f, 1.0f, 0, {}}}};
        for (int j = 0; j < ramp_node::kMaxStops; ++j) {
            const std::string n = "Stop " + std::to_string(j + 1) + " ";
            const float v = j == 0 ? 0.0f : 1.0f; // black at 0, white at 1
            ramp.params.push_back({n + "position", 0.0f, 1.0f, v, 0.01f, 4, {}});
            ramp.params.push_back({n + "red", 0.0f, 1.0f, v, 0.01f, 4, {}});
            ramp.params.push_back({n + "green", 0.0f, 1.0f, v, 0.01f, 4, {}});
            ramp.params.push_back({n + "blue", 0.0f, 1.0f, v, 0.01f, 4, {}});
        }
        s.push_back(std::move(ramp));
    }
    s.push_back({NodeKind::Color,
                 "Color",
                 "A fixed color (sRGB components in [0, 1]), the same in every cell.",
                 {},
                 {{"Color", T::Color}},
                 {{"Red", 0.0f, 1.0f, 1.0f, 0.01f, 4, {}},
                  {"Green", 0.0f, 1.0f, 1.0f, 0.01f, 4, {}},
                  {"Blue", 0.0f, 1.0f, 1.0f, 0.01f, 4, {}}}});
    s.push_back({NodeKind::Mix,
                 "Mix",
                 "Blends B into A by Factor (clamped to [0, 1]): Mix A + (B − A)·f, Add A + B·f, Multiply "
                 "A·(1 + (B − 1)·f), Screen 1 − (1 − A)(1 − B·f); the result is clamped to [0, 1]. Unlinked "
                 "colors are black. Link a field or an Oscillator into Factor to vary it per cell or over time.",
                 {{"A", T::Color, -1}, {"B", T::Color, -1}, {"Factor", T::Cell, mix_node::Factor}},
                 {{"Color", T::Color}},
                 {{"Mode", 0.0f, 3.0f, 0.0f, 1.0f, 0, {"Mix", "Add", "Multiply", "Screen"}},
                  {"Factor", 0.0f, 1.0f, 0.5f, 0.05f, 3, {}}}});
    s.push_back({NodeKind::View,
                 "View",
                 "What the viewport shows. Linked, the color wired here, for every cell, while running or "
                 "paused. Unlinked, the Simulation panel's View settings (field, palette, low, high), as before "
                 "display graphs existed.",
                 {{"Color", T::Color, -1}},
                 {},
                 {}});

    // The second state field.
    s.push_back({NodeKind::Nutrient,
                 "Nutrient N",
                 "The cell's current nutrient N, the second state field (Nutrient view). Read it to make "
                 "growth depend on it; write its next value through the Simulation node's Next nutrient. "
                 "Every cell's next A and next N come from the same current A, N and U. The Field output is "
                 "the whole nutrient, for a Convolve node.",
                 {},
                 {{"N", T::Cell}, {"Field", T::Field}},
                 {}});

    // Reusable convolution kernels.
    {
        NodeSpec k{NodeKind::ConvKernel,
                   "Kernel K",
                   "A kernel of its own, independent of the Kernel panel, for Convolve nodes (link it into "
                   "several to reuse it). Rings of the chosen core within Radius, weights innermost first, then "
                   "normalized. Shift moves where a cell looks: with Shift X = 3 a cell gathers the field around "
                   "the point 3 cells to its right (+Y is down), making the kernel asymmetric. The kernel grid "
                   "is at most 65 × 65 cells: Radius + the larger shift ≤ 32.",
                   {},
                   {{"K", T::ConvKernel}},
                   {{"Radius (cells)", 1.0f, static_cast<float>(conv_kernel_node::kMaxRadius), 8.0f, 1.0f, 0, {}},
                    {"Core", 0.0f, 3.0f, 0.0f, 1.0f, 0, {"Exponential bump", "Polynomial", "Step", "Gaussian bell"}},
                    {"Rings", 1.0f, static_cast<float>(conv_kernel_node::kMaxRings), 1.0f, 1.0f, 0, {}},
                    {"Ring 1 weight", -1.0f, 1.0f, 1.0f, 0.05f, 3, {}},
                    {"Ring 2 weight", -1.0f, 1.0f, 0.5f, 0.05f, 3, {}},
                    {"Ring 3 weight", -1.0f, 1.0f, 0.25f, 0.05f, 3, {}},
                    {"Normalization", 0.0f, 3.0f, 1.0f, 1.0f, 0, {"None (raw)", "Sum = 1", "Positive = 1", "|Sum| = 1"}},
                    {"Shift X (cells)", -16.0f, 16.0f, 0.0f, 0.5f, 2, {}},
                    {"Shift Y (cells)", -16.0f, 16.0f, 0.0f, 0.5f, 2, {}}}};
        s.push_back(std::move(k));
    }
    s.push_back({NodeKind::Convolve,
                 "Convolve",
                 "U = K ∗ field per cell: the state field linked into Field (a Density A or Nutrient N node's "
                 "Field output; unlinked = the density) weighted by the Kernel K linked into Kernel, around "
                 "each cell (the world wraps). Several Convolve nodes give a rule several neighborhoods, e.g. a "
                 "short and a long range. Select the node to preview its U in the view's Neighborhood field.",
                 {{"Field", T::Field, -1}, {"Kernel", T::ConvKernel, -1}},
                 {{"U", T::Cell}},
                 {}});
    return s;
}

// Parameters that hold whole numbers: enums and the Life masks.
bool integral(const ParamSpec& p) { return !p.choices.empty() || p.decimals == 0; }

const std::vector<NodeSpec>& specs() {
    static const std::vector<NodeSpec> s = build_specs();
    return s;
}

double out_value(const Graph& g, int node, int port, std::uint64_t gen, const GrowthParams& panel, int depth);

double in_value(const Graph& g, const Node& n, int port, std::uint64_t gen, const GrowthParams& panel, int depth) {
    const std::vector<PortSpec>& inputs = node_spec(n.kind).inputs;
    if (port < 0 || port >= static_cast<int>(inputs.size()) || inputs[port].type != PortType::Value) return 0.0;
    const int param = inputs[port].param;
    const double base = param >= 0 ? static_cast<double>(n.params[static_cast<std::size_t>(param)]) : 0.0;
    const Link* l = g.link_into(n.id, port);
    if (!l) return param >= 0 ? base : static_cast<double>(gen);
    if (depth > kMaxDepth) return base;
    return base + out_value(g, l->from_node, l->from_port, gen, panel, depth + 1);
}

double out_value(const Graph& g, int node, int port, std::uint64_t gen, const GrowthParams& panel, int depth) {
    const Node* n = g.node(node);
    if (!n) return 0.0;
    const auto p = [&](int i) { return static_cast<double>(n->params[static_cast<std::size_t>(i)]); };
    if (n->kind == NodeKind::Growth) {
        return port == growth_node::DtOut ? static_cast<double>(node_growth(g, node, panel).dt) : 0.0;
    }
    if (port != 0) return 0.0;
    switch (n->kind) {
    case NodeKind::Time:
        return p(time_node::Start) + p(time_node::Speed) * static_cast<double>(gen);
    case NodeKind::Oscillator: {
        using namespace oscillator_node;
        const double t = in_value(g, *n, TimeIn, gen, panel, depth);
        const double x = t / std::max(p(Period), 1e-6) + p(Phase);
        return in_value(g, *n, OffsetIn, gen, panel, depth) +
               in_value(g, *n, AmplitudeIn, gen, panel, depth) *
                   waveform(static_cast<Waveform>(static_cast<int>(n->params[Wave])), x);
    }
    case NodeKind::Constant:
        return p(constant_node::Value);
    default:
        return 0.0;
    }
}

} // namespace

const NodeSpec& node_spec(NodeKind kind) { return specs()[static_cast<std::size_t>(kind)]; }

const std::vector<NodeKind>& node_kinds() {
    static const std::vector<NodeKind> k = {
        NodeKind::Time,     NodeKind::Oscillator, NodeKind::Constant,     NodeKind::Kernel,       NodeKind::ConvKernel,
        NodeKind::Environment, NodeKind::Output,  NodeKind::Density,      NodeKind::Nutrient, NodeKind::Neighborhood,
        NodeKind::Convolve,
        NodeKind::Growth,   NodeKind::Add,        NodeKind::Multiply,     NodeKind::Clamp,
        NodeKind::Compare,  NodeKind::Select,     NodeKind::Life,         NodeKind::Fields,
        NodeKind::ColorRamp, NodeKind::Color,     NodeKind::Mix,          NodeKind::View};
    return k;
}

bool in_domain(NodeKind kind, GraphDomain domain) {
    switch (kind) {
    case NodeKind::Time:
    case NodeKind::Oscillator:
    case NodeKind::Constant: return true;
    case NodeKind::Fields:
    case NodeKind::ColorRamp:
    case NodeKind::Color:
    case NodeKind::Mix:
    case NodeKind::View: return domain == GraphDomain::Display;
    default: return domain == GraphDomain::Simulation;
    }
}

NodeKind sink_kind(GraphDomain domain) { return domain == GraphDomain::Display ? NodeKind::View : NodeKind::Output; }

bool is_sink(NodeKind kind) { return kind == NodeKind::Output || kind == NodeKind::View; }

const std::vector<NodeKind>& node_kinds(GraphDomain domain) {
    const auto filter = [](GraphDomain d) {
        std::vector<NodeKind> out;
        for (NodeKind k : node_kinds()) {
            if (in_domain(k, d)) out.push_back(k);
        }
        return out;
    };
    static const std::vector<NodeKind> simulation = filter(GraphDomain::Simulation);
    static const std::vector<NodeKind> display = filter(GraphDomain::Display);
    return domain == GraphDomain::Display ? display : simulation;
}

bool can_link(PortType from, PortType to) { return from == to || (from == PortType::Value && to == PortType::Cell); }

std::string life_rule_name(int birth_mask, int survive_mask) {
    std::string s = "B";
    for (int n = 0; n <= 8; ++n) {
        if (birth_mask & (1 << n)) s += static_cast<char>('0' + n);
    }
    s += "/S";
    for (int n = 0; n <= 8; ++n) {
        if (survive_mask & (1 << n)) s += static_cast<char>('0' + n);
    }
    return s;
}

const char* describe(ConnectResult r) {
    switch (r) {
    case ConnectResult::Ok: return "connected";
    case ConnectResult::BadPort: return "no such port";
    case ConnectResult::TypeMismatch: return "port types differ";
    case ConnectResult::Cycle: return "the link would make a node depend on itself";
    }
    return "";
}

double waveform(Waveform wave, double x) {
    const double f = x - std::floor(x);
    switch (wave) {
    case Waveform::Sine: return std::sin(kTwoPi * f);
    case Waveform::Triangle: return f < 0.25 ? 4.0 * f : (f < 0.75 ? 2.0 - 4.0 * f : 4.0 * f - 4.0);
    case Waveform::Square: return f < 0.5 ? 1.0 : -1.0;
    case Waveform::Saw: return 2.0 * f - 1.0;
    }
    return 0.0;
}

// ---------------------------------------------------------------------------------------------
// Graph

int Graph::add_node(NodeKind kind, float x, float y) {
    if (!in_domain(kind, domain_)) return -1;
    if (is_sink(kind) && output_node() >= 0) return output_node();
    Node n;
    n.id = next_id_++;
    n.kind = kind;
    for (const ParamSpec& p : node_spec(kind).params) n.params.push_back(p.def);
    n.x = x;
    n.y = y;
    nodes_.push_back(std::move(n));
    return nodes_.back().id;
}

bool Graph::remove_node(int id) {
    const auto it = std::find_if(nodes_.begin(), nodes_.end(), [&](const Node& n) { return n.id == id; });
    if (it == nodes_.end() || is_sink(it->kind)) return false;
    nodes_.erase(it);
    std::erase_if(links_, [&](const Link& l) { return l.from_node == id || l.to_node == id; });
    return true;
}

const Node* Graph::node(int id) const {
    const auto it = std::find_if(nodes_.begin(), nodes_.end(), [&](const Node& n) { return n.id == id; });
    return it == nodes_.end() ? nullptr : &*it;
}

Node* Graph::node(int id) { return const_cast<Node*>(std::as_const(*this).node(id)); }

int Graph::output_node() const {
    const NodeKind sink = sink_kind(domain_);
    for (const Node& n : nodes_) {
        if (n.kind == sink) return n.id;
    }
    return -1;
}

bool Graph::depends_on(int node, int upstream) const {
    std::vector<int> stack{node};
    std::vector<int> seen;
    while (!stack.empty()) {
        const int n = stack.back();
        stack.pop_back();
        if (n == upstream) return true;
        if (std::find(seen.begin(), seen.end(), n) != seen.end()) continue;
        seen.push_back(n);
        for (const Link& l : links_) {
            if (l.to_node == n) stack.push_back(l.from_node);
        }
    }
    return false;
}

ConnectResult Graph::connect(int from_node, int from_port, int to_node, int to_port) {
    const Node* from = node(from_node);
    const Node* to = node(to_node);
    if (!from || !to) return ConnectResult::BadPort;
    const NodeSpec& fs = node_spec(from->kind);
    const NodeSpec& ts = node_spec(to->kind);
    if (from_port < 0 || from_port >= static_cast<int>(fs.outputs.size()) || to_port < 0 ||
        to_port >= static_cast<int>(ts.inputs.size())) {
        return ConnectResult::BadPort;
    }
    if (!can_link(fs.outputs[static_cast<std::size_t>(from_port)].type, ts.inputs[static_cast<std::size_t>(to_port)].type)) {
        return ConnectResult::TypeMismatch;
    }
    if (depends_on(from_node, to_node)) return ConnectResult::Cycle;
    disconnect(to_node, to_port);
    links_.push_back({from_node, from_port, to_node, to_port});
    return ConnectResult::Ok;
}

bool Graph::disconnect(int to_node, int to_port) {
    return std::erase_if(links_, [&](const Link& l) { return l.to_node == to_node && l.to_port == to_port; }) > 0;
}

const Link* Graph::link_into(int to_node, int to_port) const {
    const auto it = std::find_if(links_.begin(), links_.end(),
                                 [&](const Link& l) { return l.to_node == to_node && l.to_port == to_port; });
    return it == links_.end() ? nullptr : &*it;
}

void Graph::set_param(int id, int index, float value) {
    Node* n = node(id);
    if (!n || index < 0 || index >= static_cast<int>(n->params.size())) return;
    const ParamSpec& p = node_spec(n->kind).params[static_cast<std::size_t>(index)];
    if (!std::isfinite(value)) value = p.def;
    value = std::clamp(value, p.min, p.max);
    if (integral(p)) value = std::round(value);
    n->params[static_cast<std::size_t>(index)] = value;
}

void Graph::move_node(int id, float x, float y) {
    if (Node* n = node(id)) {
        n->x = x;
        n->y = y;
    }
}

std::string Graph::assign(const std::vector<Node>& nodes, const std::vector<Link>& links) {
    Graph g(domain_);
    int outputs = 0;
    for (const Node& n : nodes) {
        const std::string where = "node " + std::to_string(n.id);
        if (n.id <= 0) return where + ": ids must be positive";
        if (g.node(n.id)) return where + ": duplicate id";
        if (std::find(node_kinds().begin(), node_kinds().end(), n.kind) == node_kinds().end()) {
            return where + ": unknown kind";
        }
        if (!in_domain(n.kind, domain_)) {
            return where + ": " + node_spec(n.kind).name +
                   (domain_ == GraphDomain::Display ? " is not a display graph node" : " is a display graph node");
        }
        if (!std::isfinite(n.x) || !std::isfinite(n.y)) return where + ": position is not finite";
        const NodeSpec& spec = node_spec(n.kind);
        if (n.params.size() != spec.params.size()) {
            return where + ": " + spec.name + " takes " + std::to_string(spec.params.size()) + " parameters, got " +
                   std::to_string(n.params.size());
        }
        for (std::size_t i = 0; i < n.params.size(); ++i) {
            const ParamSpec& p = spec.params[i];
            const float v = n.params[i];
            if (!std::isfinite(v) || v < p.min || v > p.max) return where + ": " + p.name + " is out of range";
            if (integral(p) && v != std::round(v)) return where + ": " + p.name + " must be a whole number";
        }
        if (is_sink(n.kind)) ++outputs;
        g.nodes_.push_back(n);
        g.next_id_ = std::max(g.next_id_, n.id + 1);
    }
    if (outputs != 1) {
        return domain_ == GraphDomain::Display ? "the display graph needs exactly one View node"
                                               : "the graph needs exactly one Simulation (output) node";
    }
    for (const Link& l : links) {
        const std::string where = "link " + std::to_string(l.from_node) + ":" + std::to_string(l.from_port) + " -> " +
                                  std::to_string(l.to_node) + ":" + std::to_string(l.to_port);
        if (g.link_into(l.to_node, l.to_port)) return where + ": the input already has a link";
        const ConnectResult r = g.connect(l.from_node, l.from_port, l.to_node, l.to_port);
        if (r != ConnectResult::Ok) return where + ": " + describe(r);
    }
    *this = std::move(g);
    return {};
}

// ---------------------------------------------------------------------------------------------
// Evaluation

double output_value(const Graph& graph, int node, int port, std::uint64_t generation, const GrowthParams& panel) {
    return out_value(graph, node, port, generation, panel, 0);
}

double input_value(const Graph& graph, int node, int port, std::uint64_t generation, const GrowthParams& panel) {
    const Node* n = graph.node(node);
    return n ? in_value(graph, *n, port, generation, panel, 0) : 0.0;
}

GrowthParams node_growth(const Graph& graph, int node, const GrowthParams& panel) {
    const Node* n = graph.node(node);
    if (!n || n->kind != NodeKind::Growth) return panel;
    using namespace growth_node;
    if (static_cast<int>(n->params[Source]) == Panel) return panel;
    GrowthParams g;
    g.function = static_cast<GrowthFunction>(static_cast<int>(n->params[Function]));
    g.mu = n->params[Mu];
    g.sigma = n->params[Sigma];
    g.dt = n->params[Dt];
    return g;
}

KernelWarp node_kernel(const Graph& graph, int node, std::uint64_t generation) {
    const Node* n = graph.node(node);
    if (!n || n->kind != NodeKind::Kernel) return {};
    using namespace kernel_node;
    const GrowthParams none;
    KernelWarp w;
    w.radius_delta = static_cast<float>(in_value(graph, *n, RadiusDelta, generation, none, 0));
    w.peak_shift = static_cast<float>(in_value(graph, *n, PeakShift, generation, none, 0));
    w.sharpness = static_cast<float>(in_value(graph, *n, Sharpness, generation, none, 0));
    // Same limits as the parameters, so a linked oscillator cannot collapse the kernel.
    const std::vector<ParamSpec>& p = node_spec(NodeKind::Kernel).params;
    w.radius_delta = std::clamp(w.radius_delta, p[RadiusDelta].min, p[RadiusDelta].max);
    w.peak_shift = std::clamp(w.peak_shift, p[PeakShift].min, p[PeakShift].max);
    w.sharpness = std::clamp(w.sharpness, p[Sharpness].min, p[Sharpness].max);
    return w;
}

Environment node_environment(const Graph& graph, int node, std::uint64_t generation) {
    const Node* n = graph.node(node);
    if (!n || n->kind != NodeKind::Environment) return {};
    using namespace environment_node;
    const GrowthParams none;
    Environment e;
    e.pattern = static_cast<EnvironmentPattern>(static_cast<int>(n->params[environment_node::Pattern]));
    e.strength = std::clamp(static_cast<float>(in_value(graph, *n, StrengthIn, generation, none, 0)), -1.0f, 1.0f);
    e.scale = n->params[Scale];
    e.angle = static_cast<float>(in_value(graph, *n, AngleIn, generation, none, 0) * kDegree);
    e.offset = static_cast<float>(in_value(graph, *n, OffsetIn, generation, none, 0));
    return e;
}

ConvKernel node_conv_kernel(const Graph& graph, int node) {
    ConvKernel k;
    const Node* n = graph.node(node);
    if (!n || n->kind != NodeKind::ConvKernel) return k;
    using namespace conv_kernel_node;
    const auto p = [&](int i) { return n->params[static_cast<std::size_t>(i)]; };
    k.shape.radius = static_cast<int>(p(Radius));
    k.shape.core = static_cast<KernelCore>(static_cast<int>(p(Core)));
    k.shape.rings.clear();
    for (int r = 0; r < static_cast<int>(p(Rings)); ++r) k.shape.rings.push_back(p(Ring1 + r));
    k.normalization = static_cast<KernelNormalization>(static_cast<int>(p(Normalization)));
    k.shift_x = p(ShiftX);
    k.shift_y = p(ShiftY);
    return k;
}

Grid make_conv_kernel(const ConvKernel& kernel) {
    const double half =
        std::ceil(kernel.shape.radius + std::max(std::fabs(kernel.shift_x), std::fabs(kernel.shift_y)) - 1e-9);
    if (!(half <= conv_kernel_node::kMaxHalfSize)) {
        throw std::invalid_argument("radius " + std::to_string(kernel.shape.radius) +
                                    " plus the shift exceeds the " + std::to_string(conv_kernel_node::kMaxHalfSize) +
                                    "-cell limit (at most 65 x 65 weights)");
    }
    return normalize_kernel(make_kernel(kernel.shape, kernel.shift_x, kernel.shift_y), kernel.normalization);
}

GraphFrame evaluate(const Graph& graph, std::uint64_t generation) {
    GraphFrame f;
    const int out = graph.output_node();
    if (out < 0 || graph.domain() != GraphDomain::Simulation) return f;
    if (const Link* l = graph.link_into(out, output_node::KernelIn)) {
        f.kernel_connected = true;
        f.kernel = node_kernel(graph, l->from_node, generation);
    }
    if (const Link* l = graph.link_into(out, output_node::EnvironmentIn)) {
        f.environment_connected = true;
        f.environment = node_environment(graph, l->from_node, generation);
    }
    f.rule_connected = graph.link_into(out, output_node::NextIn) != nullptr;
    f.nutrient_connected = graph.link_into(out, output_node::NutrientIn) != nullptr;
    return f;
}

// ---------------------------------------------------------------------------------------------
// Presets

bool wire_lenia_rule(Graph& g, float x, float y) {
    const int out = g.output_node();
    if (g.domain() != GraphDomain::Simulation || out < 0 || g.link_into(out, output_node::NextIn)) return false;
    int env = -1;
    if (const Link* l = g.link_into(out, output_node::EnvironmentIn)) {
        env = l->from_node;
        g.disconnect(out, output_node::EnvironmentIn); // a rule ignores it; the Field feeds the sum
    }
    constexpr float kColumn = 230.0f;
    const int neighborhood = g.add_node(NodeKind::Neighborhood, x, y);
    const int density = g.add_node(NodeKind::Density, x, y + 110.0f);
    const int growth = g.add_node(NodeKind::Growth, x + kColumn, y - 20.0f);
    g.connect(neighborhood, 0, growth, growth_node::UIn);
    float column = x + 2.0f * kColumn;

    // rate = G(U) (+ Strength × E), as lenia::step_lenia sums it.
    int rate = growth;
    int rate_port = growth_node::GOut;
    if (env >= 0) {
        const int sum = g.add_node(NodeKind::Add, column, y);
        g.connect(growth, growth_node::GOut, sum, add_node::A);
        g.connect(env, environment_node::FieldOut, sum, add_node::B);
        rate = sum;
        rate_port = 0;
        column += kColumn;
    }
    const int scaled = g.add_node(NodeKind::Multiply, column, y);
    g.connect(rate, rate_port, scaled, multiply_node::A);
    g.connect(growth, growth_node::DtOut, scaled, multiply_node::B);
    column += kColumn;
    const int next = g.add_node(NodeKind::Add, column, y + 60.0f);
    g.connect(density, 0, next, add_node::A);
    g.connect(scaled, 0, next, add_node::B);
    column += kColumn;
    const int clamp = g.add_node(NodeKind::Clamp, column, y + 60.0f); // Low 0, High 1 by default
    g.connect(next, 0, clamp, clamp_node::X);
    column += kColumn;
    g.connect(clamp, 0, out, output_node::NextIn);
    g.move_node(out, column, y - 120.0f);
    return true;
}

bool wire_lenia_rule(Graph& g) {
    float x = 20.0f, y = 0.0f;
    bool first = true;
    for (const Node& n : g.nodes()) {
        x = first ? n.x : std::min(x, n.x);
        y = first ? n.y : std::max(y, n.y);
        first = false;
    }
    // Below the lowest node's top; the tallest node (Environment) is about 215 high.
    return wire_lenia_rule(g, x, y + 260.0f);
}

bool is_life_preset(GraphPreset preset) {
    return preset == GraphPreset::GameOfLife || preset == GraphPreset::GameOfLifeCompare;
}

Graph make_preset(GraphPreset preset) {
    Graph g;
    if (preset == GraphPreset::GameOfLife) {
        const int density = g.add_node(NodeKind::Density, 20, 0);
        const int neighborhood = g.add_node(NodeKind::Neighborhood, 20, 100);
        const int life = g.add_node(NodeKind::Life, 270, 20); // B3/S23 by default
        const int out = g.add_node(NodeKind::Output, 520, 0);
        g.connect(density, 0, life, life_node::StateIn);
        g.connect(neighborhood, 0, life, life_node::NeighborsIn);
        g.connect(life, 0, out, output_node::NextIn);
        return g;
    }
    if (preset == GraphPreset::GameOfLifeCompare) {
        // Alive: next = (U = 2) + (U = 3); dead: next = (U = 3). The two tests never hold
        // together, so the sum is 0 or 1.
        const int density = g.add_node(NodeKind::Density, 20, 0);
        const int neighborhood = g.add_node(NodeKind::Neighborhood, 20, 160);
        const int two = g.add_node(NodeKind::Compare, 270, 110);
        const int three = g.add_node(NodeKind::Compare, 270, 260);
        const int either = g.add_node(NodeKind::Add, 520, 130);
        const int select = g.add_node(NodeKind::Select, 770, 0);
        const int out = g.add_node(NodeKind::Output, 1020, 0);
        for (const auto& [id, count] : {std::pair{two, 2.0f}, std::pair{three, 3.0f}}) {
            g.set_param(id, compare_node::Op, static_cast<float>(CompareOp::Equal));
            g.set_param(id, compare_node::B, count);
            g.connect(neighborhood, 0, id, compare_node::AIn);
        }
        g.connect(two, 0, either, add_node::A);
        g.connect(three, 0, either, add_node::B);
        g.connect(density, 0, select, select_node::Condition);
        g.connect(either, 0, select, select_node::IfTrue);
        g.connect(three, 0, select, select_node::IfFalse);
        g.connect(select, 0, out, output_node::NextIn);
        return g;
    }
    if (preset == GraphPreset::NutrientTrail) {
        // Density row: clamp(A + dt × (G(U) + starvation × (N − 1)), 0, 1). At N = 1 the
        // starvation term is exactly 0, so this is wire_lenia_rule's update bit for bit.
        const int neighborhood = g.add_node(NodeKind::Neighborhood, 20, 0);
        const int density = g.add_node(NodeKind::Density, 20, 110);
        const int nutrient = g.add_node(NodeKind::Nutrient, 20, 330);
        const int growth = g.add_node(NodeKind::Growth, 250, -20); // Source = Simulation panel
        g.connect(neighborhood, 0, growth, growth_node::UIn);
        const int deficit = g.add_node(NodeKind::Add, 250, 200); // N − 1
        g.connect(nutrient, 0, deficit, add_node::A);
        g.set_param(deficit, add_node::B, -1.0f);
        const int starve = g.add_node(NodeKind::Multiply, 480, 200);
        g.connect(deficit, 0, starve, multiply_node::A);
        g.set_param(starve, multiply_node::B, kTrailStarvation);
        const int rate = g.add_node(NodeKind::Add, 710, 0);
        g.connect(growth, growth_node::GOut, rate, add_node::A);
        g.connect(starve, 0, rate, add_node::B);
        const int scaled = g.add_node(NodeKind::Multiply, 940, 0);
        g.connect(rate, 0, scaled, multiply_node::A);
        g.connect(growth, growth_node::DtOut, scaled, multiply_node::B);
        const int next = g.add_node(NodeKind::Add, 1170, 60);
        g.connect(density, 0, next, add_node::A);
        g.connect(scaled, 0, next, add_node::B);
        const int clamp = g.add_node(NodeKind::Clamp, 1400, 60);
        g.connect(next, 0, clamp, clamp_node::X);

        // Nutrient row: clamp(N + recovery × (1 − N) − uptake × A × N, 0, 1).
        const int eaten = g.add_node(NodeKind::Multiply, 250, 420); // A × N
        g.connect(density, 0, eaten, multiply_node::A);
        g.connect(nutrient, 0, eaten, multiply_node::B);
        const int uptake = g.add_node(NodeKind::Multiply, 480, 420); // − uptake × A × N
        g.connect(eaten, 0, uptake, multiply_node::A);
        g.set_param(uptake, multiply_node::B, -kTrailUptake);
        const int lost = g.add_node(NodeKind::Multiply, 250, 580); // − recovery × N
        g.connect(nutrient, 0, lost, multiply_node::A);
        g.set_param(lost, multiply_node::B, -kTrailRecovery);
        const int recovery = g.add_node(NodeKind::Add, 480, 580); // recovery × (1 − N)
        g.connect(lost, 0, recovery, add_node::A);
        g.set_param(recovery, add_node::B, kTrailRecovery);
        const int regrown = g.add_node(NodeKind::Add, 710, 480);
        g.connect(nutrient, 0, regrown, add_node::A);
        g.connect(recovery, 0, regrown, add_node::B);
        const int consumed = g.add_node(NodeKind::Add, 940, 480);
        g.connect(regrown, 0, consumed, add_node::A);
        g.connect(uptake, 0, consumed, add_node::B);
        const int nutrient_clamp = g.add_node(NodeKind::Clamp, 1170, 480);
        g.connect(consumed, 0, nutrient_clamp, clamp_node::X);

        const int out = g.add_node(NodeKind::Output, 1630, 220);
        g.connect(clamp, 0, out, output_node::NextIn);
        g.connect(nutrient_clamp, 0, out, output_node::NutrientIn);
        return g;
    }
    if (preset == GraphPreset::ShortLongRange) {
        // next A = clamp(A + (S > w × L ? dt : −dt), 0, 1), S and L the two Convolve outputs.
        const int density = g.add_node(NodeKind::Density, 20, 0);
        const auto kernel = [&](int radius, float y) {
            using namespace conv_kernel_node;
            const int k = g.add_node(NodeKind::ConvKernel, 20, y);
            g.set_param(k, Radius, static_cast<float>(radius));
            g.set_param(k, Core, static_cast<float>(KernelCore::GaussianBell));
            g.set_param(k, Rings, 1.0f);
            g.set_param(k, Ring1, 1.0f);
            g.set_param(k, Normalization, static_cast<float>(KernelNormalization::Sum));
            return k;
        };
        const int near_kernel = kernel(kShortRadius, 110);
        const int far_kernel = kernel(kLongRadius, 290);
        const auto convolve = [&](int k, float y) {
            const int c = g.add_node(NodeKind::Convolve, 270, y);
            g.connect(density, field_node::FieldOut, c, convolve_node::FieldIn);
            g.connect(k, 0, c, convolve_node::KernelIn);
            return c;
        };
        const int near_u = convolve(near_kernel, 110);
        const int far_u = convolve(far_kernel, 290);
        const int weighted = g.add_node(NodeKind::Multiply, 500, 290); // w × L
        g.connect(far_u, 0, weighted, multiply_node::A);
        g.set_param(weighted, multiply_node::B, kLongWeight);
        const int wins = g.add_node(NodeKind::Compare, 730, 130); // S > w × L
        g.set_param(wins, compare_node::Op, static_cast<float>(CompareOp::Greater));
        g.connect(near_u, 0, wins, compare_node::AIn);
        g.connect(weighted, 0, wins, compare_node::BIn);
        const int rate = g.add_node(NodeKind::Select, 960, 130); // ± dt
        g.connect(wins, 0, rate, select_node::Condition);
        g.set_param(rate, select_node::IfTrue, kShortLongDt);
        g.set_param(rate, select_node::IfFalse, -kShortLongDt);
        const int next = g.add_node(NodeKind::Add, 1190, 30);
        g.connect(density, field_node::ValueOut, next, add_node::A);
        g.connect(rate, 0, next, add_node::B);
        const int clamp = g.add_node(NodeKind::Clamp, 1420, 30); // Low 0, High 1 by default
        g.connect(next, 0, clamp, clamp_node::X);
        const int out = g.add_node(NodeKind::Output, 1650, 0);
        g.connect(clamp, 0, out, output_node::NextIn);
        return g;
    }

    const int time = g.add_node(NodeKind::Time, 20, 60);
    const int osc = g.add_node(NodeKind::Oscillator, 250, 0);
    const int kernel = g.add_node(NodeKind::Kernel, 520, 0);
    const int env = g.add_node(NodeKind::Environment, 520, 230);
    const int out = g.add_node(NodeKind::Output, 800, 120);
    g.connect(time, 0, osc, oscillator_node::TimeIn);
    g.connect(kernel, 0, out, output_node::KernelIn);
    g.connect(env, 0, out, output_node::EnvironmentIn);
    g.set_param(env, environment_node::Strength, 0.0f);

    switch (preset) {
    case GraphPreset::Idle:
        break;
    case GraphPreset::BreathingKernel: {
        // Slow radius breathing (±12 %) and a faster ring-peak sway, both 0 at generation 0.
        g.set_param(osc, oscillator_node::Period, 400.0f);
        g.set_param(osc, oscillator_node::Amplitude, 0.12f);
        g.connect(osc, 0, kernel, kernel_node::RadiusDelta);
        const int sway = g.add_node(NodeKind::Oscillator, 250, 230);
        g.set_param(sway, oscillator_node::Wave, static_cast<float>(Waveform::Triangle));
        g.set_param(sway, oscillator_node::Period, 150.0f);
        g.set_param(sway, oscillator_node::Amplitude, 0.25f);
        g.connect(time, 0, sway, oscillator_node::TimeIn);
        g.connect(sway, 0, kernel, kernel_node::PeakShift);
        break;
    }
    case GraphPreset::DriftingEnvironment: {
        // Stripes drifting 0.2 cells per generation; strength pulses between 0 and 0.03,
        // starting at 0 (sine at phase 0.75 is -1).
        g.set_param(time, time_node::Speed, 0.2f);
        g.set_param(env, environment_node::Pattern, static_cast<float>(EnvironmentPattern::Stripes));
        g.set_param(env, environment_node::Scale, 96.0f);
        g.set_param(env, environment_node::Angle, 30.0f);
        g.connect(time, 0, env, environment_node::OffsetIn);
        g.set_param(osc, oscillator_node::Period, 600.0f);
        g.set_param(osc, oscillator_node::Phase, 0.75f);
        g.set_param(osc, oscillator_node::Amplitude, 0.015f);
        g.set_param(osc, oscillator_node::Offset, 0.015f);
        // The oscillator's own Time input must stay the raw generation, not the slowed Time.
        g.disconnect(osc, oscillator_node::TimeIn);
        g.connect(osc, 0, env, environment_node::StrengthIn);
        break;
    }
    case GraphPreset::GameOfLife:
    case GraphPreset::GameOfLifeCompare:
    case GraphPreset::NutrientTrail:
    case GraphPreset::ShortLongRange:
        break; // built above
    }
    wire_lenia_rule(g, 20, 480);
    return g;
}

Grid graph_kernel(const Graph& graph, const KernelShape& shape, std::uint64_t generation) {
    const GraphFrame f = evaluate(graph, generation);
    return normalize_kernel(f.kernel_connected ? make_kernel(shape, f.kernel) : make_kernel(shape),
                            KernelNormalization::Sum);
}

Grid step_graph(const Grid& world, const Graph& graph, const KernelShape& shape, const GrowthParams& growth,
                std::uint64_t generation) {
    return step_graph(world, graph, graph_kernel(graph, shape, generation), growth, generation);
}

Grid step_graph(const Grid& world, const Graph& graph, const Grid& kernel, const GrowthParams& growth,
                std::uint64_t generation) {
    const GraphFrame f = evaluate(graph, generation);
    if (f.rule_connected) {
        const RuleCompile rule = compile_rule(graph, generation, growth);
        if (!rule.error.empty()) throw std::invalid_argument(rule.error);
        return step_rule(world, kernel, rule.program);
    }
    return step_lenia(world, kernel, growth, f.environment_connected ? f.environment : Environment{});
}

WorldState step_graph(const WorldState& world, const Graph& graph, const KernelShape& shape,
                      const GrowthParams& growth, std::uint64_t generation) {
    return step_graph(world, graph, graph_kernel(graph, shape, generation), growth, generation);
}

WorldState step_graph(const WorldState& world, const Graph& graph, const Grid& kernel, const GrowthParams& growth,
                      std::uint64_t generation) {
    const GraphFrame f = evaluate(graph, generation);
    if (f.rule_connected || f.nutrient_connected) {
        const RuleCompile rule = compile_rule(graph, generation, growth);
        if (!rule.error.empty()) throw std::invalid_argument(rule.error);
        return step_rule(world, kernel, rule.program);
    }
    return {step_lenia(world.density, kernel, growth, f.environment_connected ? f.environment : Environment{}),
            world.nutrient};
}

} // namespace lenia
