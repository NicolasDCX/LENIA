#pragma once

#include "lenia/convolve.hpp"
#include "lenia/node_graph.hpp"
#include "lenia/pattern.hpp"
#include "lenia/step.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lenia {

// Per-cell rules from the node graph (docs/node_graph.md, "Cell rules"). compile_rule walks
// the nodes wired into the Output node's Next state input and emits a straight-line program:
// op i writes register i and reads only registers before it. Everything that depends on the
// generation (Value nodes linked into cell inputs, environment parameters) is folded into the
// program as constants, so a program is compiled per generation and is the same for every
// cell. The CPU interpreter (run_rule, step_rule) and the GPU one (shaders/lenia_rule.glsl,
// used by the workbench and VulkanStepper::step_rule) execute the same programs.

enum class RuleOpCode : std::uint32_t {
    Const = 0,        // k[0]
    Density = 1,      // the cell's current state A
    Neighborhood = 2, // U = (K * A) at the cell
    Growth = 3,       // growth(r[a]; mu r[b], sigma r[c], function m0)       (step.hpp)
    Add = 4,          // r[a] + r[b]
    Multiply = 5,     // r[a] * r[b]
    Clamp = 6,        // min(max(r[a], r[b]), r[c])
    Compare = 7,      // r[a] <op m0> r[b] ? 1 : 0                            (CompareOp)
    Select = 8,       // r[a] >= 0.5 ? r[b] : r[c]
    Life = 9,         // n = floor(r[b] + 0.5); bit n of (r[a] >= 0.5 ? m1 : m0), 0 unless 0 <= n <= 8
    EnvField = 10,    // k[4] * environment_field(pattern m0, scale k[0], angle k[5], offset k[3])
                      // (k[1], k[2] = cos, sin of the angle, for the GPU)
};

// 48 bytes, laid out as the GPU reads it (std430 struct of 12 scalars).
struct RuleOp {
    std::uint32_t op = 0;
    std::uint32_t a = 0, b = 0, c = 0;
    std::uint32_t m0 = 0, m1 = 0;
    float k[6] = {};
    bool operator==(const RuleOp&) const = default;
};
static_assert(sizeof(RuleOp) == 48);

// The GPU interpreter's register file holds this many values (LENIA_RULE_MAX_OPS).
constexpr std::size_t kMaxRuleOps = 64;
constexpr std::uint32_t kNoRegister = 0xffffffffu;

struct RuleProgram {
    std::vector<RuleOp> ops;
    std::uint32_t output = kNoRegister; // register holding the next state
    // Register shown as the "Growth" display field: the first Growth node's output. Without
    // one the field shows next - current.
    std::uint32_t growth = kNoRegister;
    bool operator==(const RuleProgram&) const = default;
};

struct RuleCompile {
    bool linked = false; // Next state is linked; otherwise the built-in update applies
    RuleProgram program;
    std::string error; // non-empty: the rule does not fit (program is empty)
};

// Compiles the rule at `generation`; `panel` is the Simulation panel's growth, used by Growth
// nodes with Source = panel. Fails only when the rule needs more than kMaxRuleOps ops.
RuleCompile compile_rule(const Graph& graph, std::uint64_t generation, const GrowthParams& panel);

// Next state of one cell at (x, y) of a width x height world, given its density and
// neighborhood; `growth_display`, if given, receives the Growth display value. Non-finite
// results become 0 (so do the GPU's), keeping a NaN from spreading through the world.
float run_rule(const RuleProgram& program, float density, float neighborhood, std::size_t x, std::size_t y,
               std::size_t width, std::size_t height, float* growth_display = nullptr);

// One generation of `program`: U = convolve(world, kernel), then run_rule for every cell.
Grid step_rule(const Grid& world, const Grid& kernel, const RuleProgram& program);

// The program as the GPU buffer holds it: count, output, growth, 0, then 12 words per op.
std::vector<std::uint32_t> pack_rule(const RuleProgram& program);
constexpr std::size_t kRuleBufferWords = 4 + 12 * kMaxRuleOps;

// Raw Moore neighborhood: 3x3, weight 1 on the 8 neighbors, 0 in the middle, unnormalized, so
// U is the number of live neighbors of a binary world.
Grid moore_kernel();
// Conway's glider (Richard K. Guy, 1969), 3x3, heading +x +y (right and down):
//   . # .
//   . . #
//   # # #
Pattern glider_pattern();

} // namespace lenia
