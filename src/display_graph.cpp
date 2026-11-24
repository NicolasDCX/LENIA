#include "lenia/display_graph.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

namespace lenia {

namespace {

using Op = DisplayOpCode;
using Rgba = std::array<float, 4>;

// Viridis and Inferno: the ten evenly spaced key colors of the matplotlib colormaps.
// RedBlue: ColorBrewer RdBu reversed (blue low, white middle, red high), for signed fields.
const std::vector<std::uint32_t> kKeyColors[kDisplayPaletteCount] = {
    {0x000000, 0xffffff},
    {0x440154, 0x482878, 0x3e4989, 0x31688e, 0x26828e, 0x1f9e89, 0x35b779, 0x6ece58, 0xb5de2b, 0xfde725},
    {0x000004, 0x1b0c41, 0x4a0c6b, 0x781c6d, 0xa52c60, 0xcf4446, 0xed6925, 0xfb9b06, 0xf7d13d, 0xfcffa4},
    {0x2166ac, 0x67a9cf, 0xd1e5f0, 0xf7f7f7, 0xfddbc7, 0xef8a62, 0xb2182b},
    {0x9be8ff, 0x2f8fd8, 0x16306b, 0x000000, 0x6b1c08, 0xd8562a, 0xffe39b},
};

float channel(std::uint32_t rgb, int c) { return static_cast<float>((rgb >> (16 - 8 * c)) & 0xffu) / 255.0f; }

// t clamped to [0, 1]; NaN and negative values give 0. workbench_view.frag does the same.
float unit(float t) {
    if (!(t >= 0.0f)) return 0.0f;
    return t > 1.0f ? 1.0f : t;
}

// Ramp lookup over sorted stops (position, r, g, b); workbench_view.frag's display_ramp.
std::array<float, 3> ramp_at(const Rgba* s, std::size_t n, float t) {
    if (n == 0) return {0.0f, 0.0f, 0.0f};
    if (!(t > s[0][0])) return {s[0][1], s[0][2], s[0][3]};
    for (std::size_t j = 1; j < n; ++j) {
        if (t <= s[j][0]) {
            const float d = s[j][0] - s[j - 1][0];
            const float f = d > 0.0f ? (t - s[j - 1][0]) / d : 1.0f;
            std::array<float, 3> out{};
            for (int c = 0; c < 3; ++c) out[c] = s[j - 1][c + 1] + (s[j][c + 1] - s[j - 1][c + 1]) * f;
            return out;
        }
    }
    return {s[n - 1][1], s[n - 1][2], s[n - 1][3]};
}

std::vector<Rgba> sorted_stops(std::vector<RampStop> stops) {
    std::stable_sort(stops.begin(), stops.end(),
                     [](const RampStop& a, const RampStop& b) { return a.position < b.position; });
    std::vector<Rgba> out;
    out.reserve(stops.size());
    for (const RampStop& s : stops) out.push_back({s.position, s.rgb[0], s.rgb[1], s.rgb[2]});
    return out;
}

// Emits ops for the nodes upstream of the View input, depth first, each node output once.
class Compiler {
public:
    Compiler(const Graph& g, std::uint64_t gen, DisplayProgram& program) : g_(g), gen_(gen), program_(program) {}

    // Register holding input `port` of `n`: its link, or `fallback()` if unlinked.
    template <class Fallback> std::uint32_t input(const Node& n, int port, Fallback fallback, int depth) {
        const Link* l = g_.link_into(n.id, port);
        if (!l) return fallback();
        const Node* from = g_.node(l->from_node);
        const PortType type = node_spec(from->kind).outputs[static_cast<std::size_t>(l->from_port)].type;
        if (type == PortType::Value) {
            return scalar(static_cast<float>(output_value(g_, l->from_node, l->from_port, gen_)));
        }
        return output(*from, l->from_port, depth + 1);
    }

private:
    std::uint32_t emit(const DisplayOp& op) {
        if (program_.ops.size() >= kMaxDisplayOps) throw std::length_error("too many ops");
        program_.ops.push_back(op);
        return static_cast<std::uint32_t>(program_.ops.size() - 1);
    }

    std::uint32_t constant(float r, float g, float b) {
        DisplayOp op;
        op.op = static_cast<std::uint32_t>(Op::Const);
        op.k[0] = r;
        op.k[1] = g;
        op.k[2] = b;
        op.k[3] = 1.0f;
        return emit(op);
    }

    std::uint32_t scalar(float v) { return constant(v, v, v); }

    std::uint32_t black() {
        if (black_ == kNoDisplayRegister) black_ = constant(0.0f, 0.0f, 0.0f);
        return black_;
    }

    std::uint32_t field(int f) {
        std::uint32_t& reg = fields_[static_cast<std::size_t>(f)];
        if (reg == kNoDisplayRegister) {
            DisplayOp op;
            op.op = static_cast<std::uint32_t>(Op::Field);
            op.m0 = static_cast<std::uint32_t>(f);
            reg = emit(op);
        }
        return reg;
    }

    std::uint32_t output(const Node& n, int port, int depth) {
        if (depth > static_cast<int>(kMaxDisplayOps)) throw std::length_error("chain too long");
        const std::pair<int, int> key{n.id, port};
        if (const auto it = memo_.find(key); it != memo_.end()) return it->second;

        const auto p = [&](int i) { return n.params[static_cast<std::size_t>(i)]; };
        std::uint32_t reg = kNoDisplayRegister;
        switch (n.kind) {
        case NodeKind::Fields:
            reg = field(port);
            break;
        case NodeKind::ColorRamp: {
            using namespace ramp_node;
            const std::uint32_t v = input(n, ValueIn, [&] { return field(static_cast<int>(DisplayField::Density)); }, depth);
            const std::vector<Rgba> stops = sorted_stops(ramp_stops(n));
            if (program_.stops.size() + stops.size() > kMaxDisplayStops) throw std::length_error("too many stops");
            DisplayOp op;
            op.op = static_cast<std::uint32_t>(Op::Ramp);
            op.a = v;
            op.m0 = static_cast<std::uint32_t>(program_.stops.size());
            op.m1 = static_cast<std::uint32_t>(stops.size());
            op.k[0] = p(Low);
            op.k[1] = p(High);
            reg = emit(op);
            program_.stops.insert(program_.stops.end(), stops.begin(), stops.end());
            break;
        }
        case NodeKind::Color:
            reg = constant(p(color_node::Red), p(color_node::Green), p(color_node::Blue));
            break;
        case NodeKind::Mix: {
            using namespace mix_node;
            const std::uint32_t a = input(n, AIn, [&] { return black(); }, depth);
            const std::uint32_t b = input(n, BIn, [&] { return black(); }, depth);
            const std::uint32_t f = input(n, FactorIn, [&] { return scalar(p(Factor)); }, depth);
            DisplayOp op;
            op.op = static_cast<std::uint32_t>(Op::Mix);
            op.a = a;
            op.b = b;
            op.c = f;
            op.m0 = static_cast<std::uint32_t>(std::lround(p(Mode)));
            reg = emit(op);
            break;
        }
        default:
            // Value outputs become constants in input(); nothing else has a cell or color output.
            throw std::logic_error("compile_display: not a display output");
        }
        memo_[key] = reg;
        return reg;
    }

    const Graph& g_;
    std::uint64_t gen_;
    DisplayProgram& program_;
    std::map<std::pair<int, int>, std::uint32_t> memo_;
    std::array<std::uint32_t, kDisplayFieldCount> fields_{kNoDisplayRegister, kNoDisplayRegister, kNoDisplayRegister,
                                                          kNoDisplayRegister};
    std::uint32_t black_ = kNoDisplayRegister;
};

} // namespace

const std::vector<std::uint32_t>& palette_key_colors(DisplayPalette palette) {
    return kKeyColors[std::clamp(static_cast<int>(palette), 0, kDisplayPaletteCount - 1)];
}

std::array<float, 3> palette_sample(DisplayPalette palette, float t) {
    t = std::clamp(std::isfinite(t) ? t : 0.0f, 0.0f, 1.0f);
    const std::vector<std::uint32_t>& stops = palette_key_colors(palette);
    const std::size_t n = stops.size();
    const float pos = t * static_cast<float>(n - 1);
    const std::size_t i = std::min(static_cast<std::size_t>(pos), n - 2);
    const float f = pos - static_cast<float>(i);
    std::array<float, 3> out{};
    for (int c = 0; c < 3; ++c) {
        const float ca = channel(stops[i], c);
        const float cb = channel(stops[i + 1], c);
        out[static_cast<std::size_t>(c)] = ca + (cb - ca) * f;
    }
    return out;
}

std::array<float, 3> panel_color(const DisplayRange& range, float value) {
    const float span = range.hi - range.lo;
    const float t = unit(span != 0.0f ? (value - range.lo) / span : 0.0f);
    const auto entry = static_cast<int>(t * static_cast<float>(kPaletteTableSize - 1) + 0.5f);
    return palette_sample(range.palette, static_cast<float>(entry) / (kPaletteTableSize - 1));
}

std::vector<RampStop> ramp_stops(const Node& ramp) {
    using namespace ramp_node;
    std::vector<RampStop> stops;
    if (ramp.kind != NodeKind::ColorRamp || ramp.params.size() != node_spec(NodeKind::ColorRamp).params.size()) {
        return stops;
    }
    const int count = std::clamp(static_cast<int>(ramp.params[Stops]), 2, kMaxStops);
    for (int j = 0; j < count; ++j) {
        const auto p = [&](int c) { return ramp.params[static_cast<std::size_t>(stop_param(j, c))]; };
        stops.push_back({p(Position), {p(Red), p(Green), p(Blue)}});
    }
    return stops;
}

bool set_ramp_stops(Graph& graph, int id, const std::vector<RampStop>& stops) {
    using namespace ramp_node;
    const Node* n = graph.node(id);
    if (!n || n->kind != NodeKind::ColorRamp) return false;
    std::vector<RampStop> s(stops.begin(), stops.begin() + std::min<std::ptrdiff_t>(stops.size(), kMaxStops));
    while (s.size() < 2) s.push_back({1.0f, {1.0f, 1.0f, 1.0f}});
    graph.set_param(id, Stops, static_cast<float>(s.size()));
    for (std::size_t j = 0; j < s.size(); ++j) {
        const int i = static_cast<int>(j);
        graph.set_param(id, stop_param(i, Position), s[j].position);
        graph.set_param(id, stop_param(i, Red), s[j].rgb[0]);
        graph.set_param(id, stop_param(i, Green), s[j].rgb[1]);
        graph.set_param(id, stop_param(i, Blue), s[j].rgb[2]);
    }
    return true;
}

std::vector<RampStop> palette_ramp(DisplayPalette palette) {
    const std::vector<std::uint32_t>& keys = palette_key_colors(palette);
    std::vector<RampStop> stops;
    for (std::size_t j = 0; j < keys.size(); ++j) {
        const float position = static_cast<float>(j) / static_cast<float>(keys.size() - 1);
        stops.push_back({position, {channel(keys[j], 0), channel(keys[j], 1), channel(keys[j], 2)}});
    }
    return stops;
}

std::array<float, 3> sample_ramp(const std::vector<RampStop>& sorted, float t) {
    std::vector<Rgba> s;
    s.reserve(sorted.size());
    for (const RampStop& r : sorted) s.push_back({r.position, r.rgb[0], r.rgb[1], r.rgb[2]});
    return ramp_at(s.data(), s.size(), std::isfinite(t) ? t : 0.0f);
}

DisplayCompile compile_display(const Graph& graph, std::uint64_t generation) {
    DisplayCompile result;
    const Node* view = graph.node(graph.output_node());
    if (graph.domain() != GraphDomain::Display || !view || !graph.link_into(view->id, view_node::ColorIn)) {
        return result;
    }
    result.linked = true;
    Compiler compiler(graph, generation, result.program);
    try {
        result.program.output =
            compiler.input(*view, view_node::ColorIn, [] { return kNoDisplayRegister; }, 0);
    } catch (const std::length_error&) {
        result.program = {};
        result.error = "the display graph needs more than " + std::to_string(kMaxDisplayOps) + " operations or " +
                       std::to_string(kMaxDisplayStops) + " ramp stops; simplify it";
    }
    return result;
}

std::array<float, 4> run_display(const DisplayProgram& program, float density, float neighborhood, float growth,
                                 float nutrient) {
    std::array<Rgba, kMaxDisplayOps> r{};
    const std::size_t count = std::min(program.ops.size(), kMaxDisplayOps);
    const std::size_t stop_count = std::min(program.stops.size(), kMaxDisplayStops);
    for (std::size_t i = 0; i < count; ++i) {
        const DisplayOp& o = program.ops[i];
        // Ops only read earlier registers; anything else reads 0, as compile_display never emits it.
        const auto reg = [&](std::uint32_t k) { return k < i ? r[k] : Rgba{}; };
        Rgba v{};
        switch (static_cast<Op>(o.op)) {
        case Op::Const: v = {o.k[0], o.k[1], o.k[2], o.k[3]}; break;
        case Op::Field: {
            const float f = o.m0 == 0u ? density : (o.m0 == 1u ? neighborhood : (o.m0 == 2u ? growth : nutrient));
            v = {f, f, f, 1.0f};
            break;
        }
        case Op::Ramp: {
            const float span = o.k[1] - o.k[0];
            const float t = unit(span != 0.0f ? (reg(o.a)[0] - o.k[0]) / span : 0.0f);
            const std::size_t first = std::min<std::size_t>(o.m0, stop_count);
            const std::size_t n = std::min<std::size_t>(o.m1, stop_count - first);
            const auto c = ramp_at(program.stops.data() + first, n, t);
            v = {c[0], c[1], c[2], 1.0f};
            break;
        }
        case Op::Mix: {
            const Rgba a = reg(o.a);
            const Rgba b = reg(o.b);
            const float f = unit(reg(o.c)[0]);
            for (int c = 0; c < 3; ++c) {
                float x = 0.0f;
                switch (o.m0) {
                case mix_node::Mix: x = a[c] + (b[c] - a[c]) * f; break;
                case mix_node::Add: x = a[c] + b[c] * f; break;
                case mix_node::Multiply: x = a[c] * (1.0f + (b[c] - 1.0f) * f); break;
                case mix_node::Screen: x = 1.0f - (1.0f - a[c]) * (1.0f - b[c] * f); break;
                default: break;
                }
                v[c] = unit(x);
            }
            v[3] = 1.0f;
            break;
        }
        }
        r[i] = v;
    }
    if (program.output >= count) return {0.0f, 0.0f, 0.0f, 1.0f};
    const Rgba& out = r[program.output];
    return {unit(out[0]), unit(out[1]), unit(out[2]), 1.0f};
}

std::vector<std::uint32_t> pack_display(const DisplayProgram& program) {
    std::vector<std::uint32_t> words(kDisplayBufferWords, 0u);
    const std::size_t count = std::min(program.ops.size(), kMaxDisplayOps);
    const std::size_t stop_count = std::min(program.stops.size(), kMaxDisplayStops);
    words[0] = static_cast<std::uint32_t>(count);
    words[1] = program.output;
    words[2] = static_cast<std::uint32_t>(stop_count);
    std::size_t w = 4;
    for (std::size_t i = 0; i < count; ++i) {
        const DisplayOp& o = program.ops[i];
        for (std::uint32_t v : {o.op, o.a, o.b, o.c, o.m0, o.m1}) words[w++] = v;
        for (float k : o.k) words[w++] = std::bit_cast<std::uint32_t>(k);
    }
    w = 4 + 12 * kMaxDisplayOps;
    for (std::size_t i = 0; i < stop_count; ++i) {
        for (float v : program.stops[i]) words[w++] = std::bit_cast<std::uint32_t>(v);
    }
    return words;
}

bool wire_panel_display(Graph& g, DisplayField field, const DisplayRange& range) {
    const int view = g.output_node();
    if (g.domain() != GraphDomain::Display || view < 0 || g.link_into(view, view_node::ColorIn)) return false;
    const Node* v = g.node(view);
    const float x = v->x;
    const float y = v->y;
    const int fields = g.add_node(NodeKind::Fields, x - 520.0f, y);
    const int ramp = g.add_node(NodeKind::ColorRamp, x - 260.0f, y);
    g.set_param(ramp, ramp_node::Low, range.lo);
    g.set_param(ramp, ramp_node::High, range.hi);
    set_ramp_stops(g, ramp, palette_ramp(range.palette));
    g.connect(fields, static_cast<int>(field), ramp, ramp_node::ValueIn);
    g.connect(ramp, 0, view, view_node::ColorIn);
    return true;
}

Graph make_display_preset(DisplayPreset preset, DisplayField field, const DisplayRange& range) {
    Graph g(GraphDomain::Display);
    if (preset == DisplayPreset::DensityNutrient) {
        // Nutrient as soil (dark brown depleted, green full); the density, black at 0, screened
        // over it, so empty cells show the soil unchanged.
        const int fields = g.add_node(NodeKind::Fields, 20, 120);
        const int soil = g.add_node(NodeKind::ColorRamp, 290, 0);
        set_ramp_stops(g, soil,
                       {{0.0f, {0.08f, 0.05f, 0.03f}}, {0.5f, {0.3f, 0.24f, 0.08f}}, {1.0f, {0.1f, 0.36f, 0.16f}}});
        const int creature = g.add_node(NodeKind::ColorRamp, 290, 170);
        set_ramp_stops(g, creature,
                       {{0.0f, {0.0f, 0.0f, 0.0f}}, {0.3f, {0.15f, 0.55f, 0.95f}}, {1.0f, {1.0f, 0.97f, 0.85f}}});
        const int mix = g.add_node(NodeKind::Mix, 560, 100);
        g.set_param(mix, mix_node::Mode, static_cast<float>(mix_node::Screen));
        g.set_param(mix, mix_node::Factor, 1.0f);
        const int view = g.add_node(NodeKind::View, 830, 120);
        g.connect(fields, fields_node::NutrientOut, soil, ramp_node::ValueIn);
        g.connect(fields, fields_node::DensityOut, creature, ramp_node::ValueIn);
        g.connect(soil, 0, mix, mix_node::AIn);
        g.connect(creature, 0, mix, mix_node::BIn);
        g.connect(mix, 0, view, view_node::ColorIn);
        return g;
    }
    if (preset != DisplayPreset::GrowthOverlay) {
        g.add_node(NodeKind::View, 560, 40);
        if (preset == DisplayPreset::WiredPanel) wire_panel_display(g, field, range);
        return g;
    }

    // Density through a deep-sea ramp; growth (blue shrinking, black steady, orange growing)
    // screened over it with a factor pulsing between 0.25 and 0.75 every 240 generations.
    const int fields = g.add_node(NodeKind::Fields, 20, 120);
    const int density = g.add_node(NodeKind::ColorRamp, 290, 0);
    set_ramp_stops(g, density,
                   {{0.0f, {0.02f, 0.03f, 0.08f}},
                    {0.35f, {0.05f, 0.25f, 0.38f}},
                    {0.7f, {0.35f, 0.75f, 0.7f}},
                    {1.0f, {0.95f, 0.98f, 0.85f}}});
    const int growth = g.add_node(NodeKind::ColorRamp, 290, 150);
    g.set_param(growth, ramp_node::Low, -1.0f);
    g.set_param(growth, ramp_node::High, 1.0f);
    set_ramp_stops(g, growth, {{0.0f, {0.15f, 0.4f, 1.0f}}, {0.5f, {0.0f, 0.0f, 0.0f}}, {1.0f, {1.0f, 0.55f, 0.1f}}});
    const int pulse = g.add_node(NodeKind::Oscillator, 290, 300);
    g.set_param(pulse, oscillator_node::Period, 240.0f);
    g.set_param(pulse, oscillator_node::Amplitude, 0.25f);
    g.set_param(pulse, oscillator_node::Offset, 0.5f);
    const int mix = g.add_node(NodeKind::Mix, 560, 120);
    g.set_param(mix, mix_node::Mode, static_cast<float>(mix_node::Screen));
    const int view = g.add_node(NodeKind::View, 830, 140);
    g.connect(fields, fields_node::DensityOut, density, ramp_node::ValueIn);
    g.connect(fields, fields_node::GrowthOut, growth, ramp_node::ValueIn);
    g.connect(density, 0, mix, mix_node::AIn);
    g.connect(growth, 0, mix, mix_node::BIn);
    g.connect(pulse, 0, mix, mix_node::FactorIn);
    g.connect(mix, 0, view, view_node::ColorIn);
    return g;
}

} // namespace lenia
