#pragma once

#include "lenia/convolve.hpp"
#include "lenia/node_graph.hpp"
#include "lenia/pattern.hpp"
#include "lenia/step.hpp"
#include "lenia/world.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
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
    Nutrient = 11,    // the cell's current nutrient N (world.hpp)
    Convolve = 12,    // program.kernels[m0] convolved with field m1 (0 density, 1 nutrient) at the
                      // cell; a = the kernel's offset in the program's packed kernels, b x c its
                      // size (for the GPU, which convolves in place)
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
// Convolve nodes: at most this many distinct (field, kernel) convolutions per rule, with at
// most kMaxConvolveWeights weights together (four 65 x 65 kernels).
constexpr std::size_t kMaxConvolutions = 4;
constexpr std::size_t kMaxConvolveWeights = 4 * 65 * 65;

struct RuleProgram {
    std::vector<RuleOp> ops;
    // Kernels of the Convolve ops (op m0 indexes this), odd-sized, already normalized.
    std::vector<Grid> kernels;
    std::uint32_t output = kNoRegister; // register holding the next state
    // Register shown as the "Growth" display field: the first Growth node's output. Without
    // one the field shows next - current.
    std::uint32_t growth = kNoRegister;
    // Register holding the next nutrient (Next nutrient's link); none: the nutrient is kept.
    std::uint32_t nutrient = kNoRegister;
    bool operator==(const RuleProgram&) const = default;
};

struct RuleCompile {
    bool linked = false; // Next state (or Next nutrient) is linked; otherwise the built-in update applies
    RuleProgram program;
    std::string error; // non-empty: the rule cannot run (program is empty)
    // Convolve node id -> the register holding its U, for the nodes the rule uses (previews).
    std::vector<std::pair<int, std::uint32_t>> convolve_registers;
};

// Compiles the rule at `generation`; `panel` is the Simulation panel's growth, used by Growth
// nodes with Source = panel. Next state and Next nutrient compile into one program, sharing
// the nodes both depend on. Convolve nodes with the same field and equal kernels share one
// convolution. Fails when the rule needs more than kMaxRuleOps ops or kMaxConvolutions
// convolutions, when Next nutrient is linked without Next state, when a Convolve node the rule
// uses has no Kernel K linked, or when a kernel cannot be built (make_conv_kernel).
RuleCompile compile_rule(const Graph& graph, std::uint64_t generation, const GrowthParams& panel);

// One cell's state as a rule reads it, and what the rule makes of it.
struct RuleCell {
    float density = 0.0f;
    float nutrient = 0.0f;
    float neighborhood = 0.0f;
    // program.kernels[i] convolved with its field at the cell (step_rule fills these).
    std::array<float, kMaxConvolutions> convolved{};
};
struct RuleResult {
    float density = 0.0f;  // next A
    float nutrient = 0.0f; // next N (the current one when the program does not write it)
    float growth = 0.0f;   // the Growth display value
};

// Next state of one cell at (x, y) of a width x height world. Non-finite results become 0 (so
// do the GPU's), keeping a NaN from spreading through the world.
RuleResult run_rule(const RuleProgram& program, const RuleCell& cell, std::size_t x, std::size_t y,
                    std::size_t width, std::size_t height);
// The density alone (nutrient 0); `growth_display`, if given, receives the Growth display value.
float run_rule(const RuleProgram& program, float density, float neighborhood, std::size_t x, std::size_t y,
               std::size_t width, std::size_t height, float* growth_display = nullptr);

// One generation of `program`: U = convolve(world, kernel), each Convolve op's kernel convolved
// with its field (a nutrient of 0 here), then run_rule for every cell.
Grid step_rule(const Grid& world, const Grid& kernel, const RuleProgram& program);
// Both state fields at once: every cell's next density and nutrient come from the current
// density, nutrient, U and Convolve results. Throws std::invalid_argument if the fields differ
// in size.
WorldState step_rule(const WorldState& world, const Grid& kernel, const RuleProgram& program);

// The program as the GPU buffer holds it: count, output, growth, nutrient, then 12 words per op.
// The Convolve ops' kernels go into the GPU's kernel buffer at `kernel_base` floats (after the
// active kernel, pack_rule_kernels), so their offsets (word a) are shifted by it.
std::vector<std::uint32_t> pack_rule(const RuleProgram& program, std::uint32_t kernel_base = 0);
constexpr std::size_t kRuleBufferWords = 4 + 12 * kMaxRuleOps;
// The Convolve kernels' weights, one after another (at most kMaxConvolveWeights).
std::vector<float> pack_rule_kernels(const RuleProgram& program);

// Raw Moore neighborhood: 3x3, weight 1 on the 8 neighbors, 0 in the middle, unnormalized, so
// U is the number of live neighbors of a binary world.
Grid moore_kernel();
// Conway's glider (Richard K. Guy, 1969), 3x3, heading +x +y (right and down):
//   . # .
//   . . #
//   # # #
Pattern glider_pattern();

} // namespace lenia
