#include "lenia/rule.hpp"

#include "lenia/environment.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

namespace lenia {

namespace {

using Op = RuleOpCode;

// Emits ops for the nodes upstream of a cell input, depth first, each node output once.
class Compiler {
public:
    Compiler(const Graph& g, std::uint64_t gen, const GrowthParams& panel, RuleProgram& program)
        : g_(g), gen_(gen), panel_(panel), program_(program) {}

    // Register holding the value of input `port` of `n`: its link, or `fallback()` if unlinked.
    template <class Fallback> std::uint32_t input(const Node& n, int port, Fallback fallback, int depth) {
        const Link* l = g_.link_into(n.id, port);
        if (!l) return fallback();
        const Node* from = g_.node(l->from_node);
        const PortType type = node_spec(from->kind).outputs[static_cast<std::size_t>(l->from_port)].type;
        if (type == PortType::Value) {
            // The same value in every cell, fixed for this generation.
            return constant(static_cast<float>(output_value(g_, l->from_node, l->from_port, gen_, panel_)));
        }
        return output(*from, l->from_port, depth + 1);
    }

    // An input with a parameter: the parameter while unlinked.
    std::uint32_t param_input(const Node& n, int port, int depth) {
        const int param = node_spec(n.kind).inputs[static_cast<std::size_t>(port)].param;
        return input(n, port, [&] { return constant(n.params[static_cast<std::size_t>(param)]); }, depth);
    }

private:
    std::uint32_t emit(const RuleOp& op) {
        if (program_.ops.size() >= kMaxRuleOps) throw std::length_error("too many ops");
        program_.ops.push_back(op);
        return static_cast<std::uint32_t>(program_.ops.size() - 1);
    }

    std::uint32_t simple(Op code, std::uint32_t a = 0, std::uint32_t b = 0, std::uint32_t c = 0) {
        RuleOp op;
        op.op = static_cast<std::uint32_t>(code);
        op.a = a;
        op.b = b;
        op.c = c;
        return emit(op);
    }

    std::uint32_t constant(float v) {
        RuleOp op;
        op.op = static_cast<std::uint32_t>(Op::Const);
        op.k[0] = v;
        return emit(op);
    }

    std::uint32_t density() {
        if (density_ == kNoRegister) density_ = simple(Op::Density);
        return density_;
    }

    std::uint32_t neighborhood() {
        if (neighborhood_ == kNoRegister) neighborhood_ = simple(Op::Neighborhood);
        return neighborhood_;
    }

    std::uint32_t output(const Node& n, int port, int depth) {
        // Every node on a chain emits at least one op, so a longer chain cannot fit anyway.
        if (depth > static_cast<int>(kMaxRuleOps)) throw std::length_error("chain too long");
        const std::pair<int, int> key{n.id, port};
        if (const auto it = memo_.find(key); it != memo_.end()) return it->second;

        std::uint32_t reg = kNoRegister;
        const auto p = [&](int i) { return n.params[static_cast<std::size_t>(i)]; };
        const auto mask = [&](int i) { return static_cast<std::uint32_t>(std::lround(p(i))); };
        switch (n.kind) {
        case NodeKind::Density:
            reg = density();
            break;
        case NodeKind::Neighborhood:
            reg = neighborhood();
            break;
        case NodeKind::Environment: {
            const Environment e = node_environment(g_, n.id, gen_);
            RuleOp op;
            op.op = static_cast<std::uint32_t>(Op::EnvField);
            op.m0 = static_cast<std::uint32_t>(e.pattern);
            op.k[0] = e.scale;
            op.k[1] = std::cos(e.angle);
            op.k[2] = std::sin(e.angle);
            op.k[3] = e.offset;
            op.k[4] = e.strength;
            op.k[5] = e.angle;
            reg = emit(op);
            break;
        }
        case NodeKind::Growth: {
            using namespace growth_node;
            const GrowthParams params = node_growth(g_, n.id, panel_);
            const std::uint32_t u = input(n, UIn, [&] { return neighborhood(); }, depth);
            const std::uint32_t mu = input(n, MuIn, [&] { return constant(params.mu); }, depth);
            const std::uint32_t sigma = input(n, SigmaIn, [&] { return constant(params.sigma); }, depth);
            RuleOp op;
            op.op = static_cast<std::uint32_t>(Op::Growth);
            op.a = u;
            op.b = mu;
            op.c = sigma;
            op.m0 = static_cast<std::uint32_t>(params.function);
            reg = emit(op);
            if (program_.growth == kNoRegister) program_.growth = reg;
            break;
        }
        case NodeKind::Add:
        case NodeKind::Multiply: {
            const std::uint32_t a = param_input(n, 0, depth);
            const std::uint32_t b = param_input(n, 1, depth);
            reg = simple(n.kind == NodeKind::Add ? Op::Add : Op::Multiply, a, b);
            break;
        }
        case NodeKind::Clamp: {
            const std::uint32_t x = param_input(n, clamp_node::X, depth);
            const std::uint32_t lo = param_input(n, clamp_node::Low, depth);
            const std::uint32_t hi = param_input(n, clamp_node::High, depth);
            reg = simple(Op::Clamp, x, lo, hi);
            break;
        }
        case NodeKind::Compare: {
            const std::uint32_t a = param_input(n, compare_node::AIn, depth);
            const std::uint32_t b = param_input(n, compare_node::BIn, depth);
            RuleOp op;
            op.op = static_cast<std::uint32_t>(Op::Compare);
            op.a = a;
            op.b = b;
            op.m0 = mask(compare_node::Op);
            reg = emit(op);
            break;
        }
        case NodeKind::Select: {
            const std::uint32_t c = param_input(n, select_node::Condition, depth);
            const std::uint32_t t = param_input(n, select_node::IfTrue, depth);
            const std::uint32_t f = param_input(n, select_node::IfFalse, depth);
            reg = simple(Op::Select, c, t, f);
            break;
        }
        case NodeKind::Life: {
            const std::uint32_t state = input(n, life_node::StateIn, [&] { return density(); }, depth);
            const std::uint32_t count = input(n, life_node::NeighborsIn, [&] { return neighborhood(); }, depth);
            RuleOp op;
            op.op = static_cast<std::uint32_t>(Op::Life);
            op.a = state;
            op.b = count;
            op.m0 = mask(life_node::Birth);
            op.m1 = mask(life_node::Survive);
            reg = emit(op);
            break;
        }
        default:
            // Only cell outputs get here; Value outputs become constants in input().
            throw std::logic_error("compile_rule: not a cell output");
        }
        memo_[key] = reg;
        return reg;
    }

    const Graph& g_;
    std::uint64_t gen_;
    const GrowthParams& panel_;
    RuleProgram& program_;
    std::map<std::pair<int, int>, std::uint32_t> memo_;
    std::uint32_t density_ = kNoRegister;
    std::uint32_t neighborhood_ = kNoRegister;
};

bool compare(std::uint32_t op, float a, float b) {
    switch (static_cast<CompareOp>(op)) {
    case CompareOp::Greater: return a > b;
    case CompareOp::GreaterEqual: return a >= b;
    case CompareOp::Less: return a < b;
    case CompareOp::LessEqual: return a <= b;
    case CompareOp::Equal: return a == b;
    case CompareOp::NotEqual: return a != b;
    }
    return false;
}

float finite_or_zero(float v) { return std::isfinite(v) ? v : 0.0f; }

} // namespace

RuleCompile compile_rule(const Graph& graph, std::uint64_t generation, const GrowthParams& panel) {
    RuleCompile result;
    const Node* out = graph.node(graph.output_node());
    if (!out || !graph.link_into(out->id, output_node::NextIn)) return result;
    result.linked = true;
    Compiler compiler(graph, generation, panel, result.program);
    try {
        result.program.output =
            compiler.input(*out, output_node::NextIn, [] { return kNoRegister; }, 0);
    } catch (const std::length_error&) {
        result.program = {};
        result.error = "the rule needs more than " + std::to_string(kMaxRuleOps) +
                       " operations (each node, constant and unlinked input parameter counts); simplify it";
    }
    return result;
}

float run_rule(const RuleProgram& program, float density, float neighborhood, std::size_t x, std::size_t y,
               std::size_t width, std::size_t height, float* growth_display) {
    std::array<float, kMaxRuleOps> r{};
    const std::size_t count = std::min(program.ops.size(), kMaxRuleOps);
    for (std::size_t i = 0; i < count; ++i) {
        const RuleOp& o = program.ops[i];
        // Ops only read earlier registers; anything else reads 0, as compile_rule never emits it.
        const auto reg = [&](std::uint32_t k) { return k < i ? r[k] : 0.0f; };
        float v = 0.0f;
        switch (static_cast<Op>(o.op)) {
        case Op::Const: v = o.k[0]; break;
        case Op::Density: v = density; break;
        case Op::Neighborhood: v = neighborhood; break;
        case Op::Growth: {
            GrowthParams g;
            g.mu = reg(o.b);
            g.sigma = reg(o.c);
            g.function = o.m0 == 1u ? GrowthFunction::Polynomial : GrowthFunction::Gaussian;
            v = growth(g, reg(o.a));
            break;
        }
        case Op::Add: v = reg(o.a) + reg(o.b); break;
        case Op::Multiply: v = reg(o.a) * reg(o.b); break;
        case Op::Clamp: v = std::min(std::max(reg(o.a), reg(o.b)), reg(o.c)); break;
        case Op::Compare: v = compare(o.m0, reg(o.a), reg(o.b)) ? 1.0f : 0.0f; break;
        case Op::Select: v = reg(o.a) >= 0.5f ? reg(o.b) : reg(o.c); break;
        case Op::Life: {
            const float n = std::floor(reg(o.b) + 0.5f);
            const std::uint32_t mask = reg(o.a) >= 0.5f ? o.m1 : o.m0;
            v = n >= 0.0f && n <= 8.0f ? static_cast<float>((mask >> static_cast<std::uint32_t>(n)) & 1u) : 0.0f;
            break;
        }
        case Op::EnvField: {
            Environment e;
            e.pattern = static_cast<EnvironmentPattern>(o.m0);
            e.scale = o.k[0];
            e.offset = o.k[3];
            e.strength = o.k[4];
            e.angle = o.k[5];
            v = e.strength * environment_field(e, x, y, width, height);
            break;
        }
        }
        r[i] = v;
    }
    const float next = finite_or_zero(program.output < count ? r[program.output] : density);
    if (growth_display) {
        *growth_display = finite_or_zero(program.growth < count ? r[program.growth] : next - density);
    }
    return next;
}

Grid step_rule(const Grid& world, const Grid& kernel, const RuleProgram& program) {
    Grid next = convolve(world, kernel); // holds U, overwritten in place below
    for (std::size_t y = 0; y < next.height; ++y) {
        for (std::size_t x = 0; x < next.width; ++x) {
            const std::size_t i = y * next.width + x;
            next.cells[i] = run_rule(program, world.cells[i], next.cells[i], x, y, next.width, next.height);
        }
    }
    return next;
}

std::vector<std::uint32_t> pack_rule(const RuleProgram& program) {
    std::vector<std::uint32_t> words;
    const std::size_t count = std::min(program.ops.size(), kMaxRuleOps);
    words.reserve(4 + 12 * count);
    words.push_back(static_cast<std::uint32_t>(count));
    words.push_back(program.output);
    words.push_back(program.growth);
    words.push_back(0u);
    for (std::size_t i = 0; i < count; ++i) {
        const RuleOp& o = program.ops[i];
        for (std::uint32_t w : {o.op, o.a, o.b, o.c, o.m0, o.m1}) words.push_back(w);
        for (float k : o.k) words.push_back(std::bit_cast<std::uint32_t>(k));
    }
    return words;
}

Grid moore_kernel() {
    Grid k(3, 3, 1.0f);
    k.at(1, 1) = 0.0f;
    return k;
}

Pattern glider_pattern() {
    Pattern p;
    p.name = "Glider";
    p.source = "Richard K. Guy, 1969; John H. Conway's Game of Life (B3/S23)";
    p.license = "Public domain (a 5-cell mathematical configuration)";
    p.cells = Grid(3, 3, 0.0f);
    p.cells.at(1, 0) = 1.0f;
    p.cells.at(2, 1) = 1.0f;
    p.cells.at(0, 2) = 1.0f;
    p.cells.at(1, 2) = 1.0f;
    p.cells.at(2, 2) = 1.0f;
    return p;
}

} // namespace lenia
