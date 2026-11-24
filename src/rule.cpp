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

// A rule that cannot run, with the message compile_rule reports.
struct CompileError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// make_conv_kernel, remembering the last few kernels: compile_rule runs before every generation,
// and rebuilding a 65 x 65 kernel each time would cost more than the rest of the compile.
Grid cached_conv_kernel(const ConvKernel& k) {
    struct Entry {
        ConvKernel key;
        Grid kernel;
    };
    constexpr std::size_t kEntries = 2 * kMaxConvolutions;
    thread_local std::vector<Entry> cache; // most recently used last
    const auto it = std::find_if(cache.begin(), cache.end(), [&](const Entry& e) { return e.key == k; });
    if (it != cache.end()) {
        std::rotate(it, it + 1, cache.end());
        return cache.back().kernel;
    }
    Grid kernel = make_conv_kernel(k); // throws std::invalid_argument
    if (cache.size() >= kEntries) cache.erase(cache.begin());
    cache.push_back({k, kernel});
    return kernel;
}

// The field each of program.kernels is convolved with (0 density, 1 nutrient), from its ops.
std::vector<std::uint32_t> kernel_fields(const RuleProgram& program) {
    std::vector<std::uint32_t> fields(program.kernels.size(), 0u);
    for (const RuleOp& o : program.ops) {
        if (o.op == static_cast<std::uint32_t>(Op::Convolve) && o.m0 < fields.size()) fields[o.m0] = o.m1;
    }
    return fields;
}

// Emits ops for the nodes upstream of a cell input, depth first, each node output once.
class Compiler {
public:
    Compiler(const Graph& g, std::uint64_t gen, const GrowthParams& panel, RuleProgram& program)
        : g_(g), gen_(gen), panel_(panel), program_(program) {}

    // Convolve node id -> the register holding its U, in compile order.
    const std::vector<std::pair<int, std::uint32_t>>& convolve_registers() const { return convolve_registers_; }

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

    std::uint32_t nutrient() {
        if (nutrient_ == kNoRegister) nutrient_ = simple(Op::Nutrient);
        return nutrient_;
    }

    // U = K ∗ field for Convolve node `n`: one convolution per distinct (field, kernel) pair,
    // shared by every Convolve node that asks for the same one.
    std::uint32_t convolve(const Node& n) {
        const std::string name = "Convolve node " + std::to_string(n.id);
        std::uint32_t field = 0; // unlinked: the density
        if (const Link* l = g_.link_into(n.id, convolve_node::FieldIn)) {
            field = g_.node(l->from_node)->kind == NodeKind::Nutrient ? 1u : 0u;
        }
        const Link* kl = g_.link_into(n.id, convolve_node::KernelIn);
        if (!kl) throw CompileError(name + " has no Kernel K linked into its Kernel input");
        Grid kernel;
        try {
            kernel = cached_conv_kernel(node_conv_kernel(g_, kl->from_node));
        } catch (const std::invalid_argument& e) {
            throw CompileError("Kernel K node " + std::to_string(kl->from_node) + ": " + e.what());
        }

        std::size_t index = 0;
        while (index < program_.kernels.size() &&
               !(fields_[index] == field && program_.kernels[index] == kernel)) {
            ++index;
        }
        if (index == program_.kernels.size()) {
            if (program_.kernels.size() >= kMaxConvolutions) {
                throw CompileError("the rule needs more than " + std::to_string(kMaxConvolutions) +
                                   " different convolutions (Convolve nodes with the same field and equal kernels "
                                   "share one); remove a Convolve node or reuse a Kernel K");
            }
            std::size_t weights = kernel.cells.size();
            for (const Grid& k : program_.kernels) weights += k.cells.size();
            if (weights > kMaxConvolveWeights) {
                throw CompileError("the rule's Convolve kernels hold more than " +
                                   std::to_string(kMaxConvolveWeights) + " weights together; make them smaller");
            }
            program_.kernels.push_back(std::move(kernel));
            fields_.push_back(field);
        }
        if (const auto it = convolution_registers_.find(index); it != convolution_registers_.end()) {
            return it->second;
        }
        std::uint32_t offset = 0;
        for (std::size_t i = 0; i < index; ++i) offset += static_cast<std::uint32_t>(program_.kernels[i].cells.size());
        RuleOp op;
        op.op = static_cast<std::uint32_t>(Op::Convolve);
        op.a = offset;
        op.b = static_cast<std::uint32_t>(program_.kernels[index].width);
        op.c = static_cast<std::uint32_t>(program_.kernels[index].height);
        op.m0 = static_cast<std::uint32_t>(index);
        op.m1 = field;
        const std::uint32_t reg = emit(op);
        convolution_registers_[index] = reg;
        return reg;
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
        case NodeKind::Nutrient:
            reg = nutrient();
            break;
        case NodeKind::Convolve:
            reg = convolve(n);
            convolve_registers_.emplace_back(n.id, reg);
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
    std::uint32_t nutrient_ = kNoRegister;
    std::vector<std::uint32_t> fields_; // the field of each of program_.kernels
    std::map<std::size_t, std::uint32_t> convolution_registers_; // kernels index -> its Convolve op
    std::vector<std::pair<int, std::uint32_t>> convolve_registers_;
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
    if (!out || graph.domain() != GraphDomain::Simulation) return result;
    const bool next = graph.link_into(out->id, output_node::NextIn) != nullptr;
    const bool nutrient = graph.link_into(out->id, output_node::NutrientIn) != nullptr;
    if (!next && !nutrient) return result;
    result.linked = true;
    if (!next) {
        result.error = "Next nutrient is linked but Next state is not: link the density's rule into Next state "
                       "too (select the Simulation node to wire the Lenia rule)";
        return result;
    }
    Compiler compiler(graph, generation, panel, result.program);
    try {
        result.program.output =
            compiler.input(*out, output_node::NextIn, [] { return kNoRegister; }, 0);
        if (nutrient) {
            result.program.nutrient = compiler.input(*out, output_node::NutrientIn, [] { return kNoRegister; }, 0);
        }
        result.convolve_registers = compiler.convolve_registers();
    } catch (const std::length_error&) {
        result.program = {};
        result.error = "the rule needs more than " + std::to_string(kMaxRuleOps) +
                       " operations (each node, constant and unlinked input parameter counts); simplify it";
    } catch (const CompileError& e) {
        result.program = {};
        result.error = e.what();
    }
    return result;
}

RuleResult run_rule(const RuleProgram& program, const RuleCell& cell, std::size_t x, std::size_t y,
                    std::size_t width, std::size_t height) {
    const float density = cell.density;
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
        case Op::Neighborhood: v = cell.neighborhood; break;
        case Op::Nutrient: v = cell.nutrient; break;
        case Op::Convolve: v = o.m0 < kMaxConvolutions ? cell.convolved[o.m0] : 0.0f; break;
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
    RuleResult out;
    out.density = finite_or_zero(program.output < count ? r[program.output] : density);
    out.nutrient = finite_or_zero(program.nutrient < count ? r[program.nutrient] : cell.nutrient);
    out.growth = finite_or_zero(program.growth < count ? r[program.growth] : out.density - density);
    return out;
}

float run_rule(const RuleProgram& program, float density, float neighborhood, std::size_t x, std::size_t y,
               std::size_t width, std::size_t height, float* growth_display) {
    const RuleResult r = run_rule(program, RuleCell{density, 0.0f, neighborhood}, x, y, width, height);
    if (growth_display) *growth_display = r.growth;
    return r.density;
}

namespace {
// Each Convolve kernel convolved with its field; `nutrient` null reads a nutrient of 0.
std::vector<Grid> convolutions(const RuleProgram& program, const Grid& density, const Grid* nutrient) {
    const std::vector<std::uint32_t> fields = kernel_fields(program);
    std::vector<Grid> out;
    const std::size_t count = std::min(program.kernels.size(), kMaxConvolutions);
    for (std::size_t i = 0; i < count; ++i) {
        if (fields[i] == 1u && !nutrient) out.emplace_back(density.width, density.height, 0.0f);
        else out.push_back(convolve(fields[i] == 1u ? *nutrient : density, program.kernels[i]));
    }
    return out;
}

RuleCell rule_cell(const std::vector<Grid>& convolved, std::size_t i, float density, float nutrient, float u) {
    RuleCell cell{density, nutrient, u};
    for (std::size_t k = 0; k < convolved.size(); ++k) cell.convolved[k] = convolved[k].cells[i];
    return cell;
}
} // namespace

Grid step_rule(const Grid& world, const Grid& kernel, const RuleProgram& program) {
    const std::vector<Grid> convolved = convolutions(program, world, nullptr);
    Grid next = convolve(world, kernel); // holds U, overwritten in place below
    for (std::size_t y = 0; y < next.height; ++y) {
        for (std::size_t x = 0; x < next.width; ++x) {
            const std::size_t i = y * next.width + x;
            next.cells[i] = run_rule(program, rule_cell(convolved, i, world.cells[i], 0.0f, next.cells[i]), x, y,
                                     next.width, next.height)
                                .density;
        }
    }
    return next;
}

WorldState step_rule(const WorldState& world, const Grid& kernel, const RuleProgram& program) {
    const Grid& a = world.density;
    if (world.nutrient.width != a.width || world.nutrient.height != a.height ||
        world.nutrient.cells.size() != a.cells.size()) {
        throw std::invalid_argument("step_rule: the nutrient and density fields differ in size");
    }
    const Grid u = convolve(a, kernel);
    const std::vector<Grid> convolved = convolutions(program, a, &world.nutrient);
    // Fresh grids for the results: every cell reads only the current state.
    WorldState next{Grid(a.width, a.height), Grid(a.width, a.height)};
    for (std::size_t y = 0; y < a.height; ++y) {
        for (std::size_t x = 0; x < a.width; ++x) {
            const std::size_t i = y * a.width + x;
            const RuleResult r = run_rule(program, rule_cell(convolved, i, a.cells[i], world.nutrient.cells[i], u.cells[i]),
                                          x, y, a.width, a.height);
            next.density.cells[i] = r.density;
            next.nutrient.cells[i] = r.nutrient;
        }
    }
    return next;
}

std::vector<std::uint32_t> pack_rule(const RuleProgram& program, std::uint32_t kernel_base) {
    std::vector<std::uint32_t> words;
    const std::size_t count = std::min(program.ops.size(), kMaxRuleOps);
    words.reserve(4 + 12 * count);
    words.push_back(static_cast<std::uint32_t>(count));
    words.push_back(program.output);
    words.push_back(program.growth);
    words.push_back(program.nutrient);
    for (std::size_t i = 0; i < count; ++i) {
        const RuleOp& o = program.ops[i];
        const std::uint32_t a = o.op == static_cast<std::uint32_t>(Op::Convolve) ? o.a + kernel_base : o.a;
        for (std::uint32_t w : {o.op, a, o.b, o.c, o.m0, o.m1}) words.push_back(w);
        for (float k : o.k) words.push_back(std::bit_cast<std::uint32_t>(k));
    }
    return words;
}

std::vector<float> pack_rule_kernels(const RuleProgram& program) {
    std::vector<float> weights;
    for (const Grid& k : program.kernels) weights.insert(weights.end(), k.cells.begin(), k.cells.end());
    return weights;
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
