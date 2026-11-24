#pragma once

#include "lenia/node_graph.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lenia {

// Display graphs (docs/display_graph.md): a Graph in the Display domain colors the viewport.
// Fields nodes read density, neighborhood and growth per cell, Color ramps map them to colors,
// Color and Mix nodes combine colors, and the View node receives the result. compile_display
// turns it into a straight-line program, the same for every cell, that the workbench's
// fragment shader (shaders/workbench_view.frag) runs per pixel; run_display is the CPU
// reference. Value nodes (Time, Oscillator, Constant) are folded in as per-generation
// constants, as in cell rules, so a display can be animated and still replay exactly.
//
// With the View node's input unlinked the viewport shows the Simulation panel's display (one
// field through a palette), as before display graphs existed.

// The panel display: field and palette choices, in the workbench's order.
enum class DisplayField { Density, Neighborhood, Growth };
enum class DisplayPalette { Grayscale, Viridis, Inferno, RedBlue, IceFire };
constexpr int kDisplayFieldCount = 3;
constexpr int kDisplayPaletteCount = 5;

struct DisplayRange {
    DisplayPalette palette = DisplayPalette::Viridis;
    float lo = 0.0f; // field value at the first palette entry
    float hi = 1.0f; // field value at the last palette entry
};

// A palette's evenly spaced key colors, 0xRRGGBB.
const std::vector<std::uint32_t>& palette_key_colors(DisplayPalette palette);
// Linear interpolation between the key colors at t in [0, 1] (clamped; non-finite = 0): the
// palette lookup tables the panel display uses.
std::array<float, 3> palette_sample(DisplayPalette palette, float t);
// Entries per palette lookup table, and the panel display's color for a field value: the
// table entry nearest to t = (value − lo) / (hi − lo) clamped (0 when hi = lo), as
// workbench_view.frag draws it.
constexpr int kPaletteTableSize = 256;
std::array<float, 3> panel_color(const DisplayRange& range, float value);

// Color ramp stops. Positions in [0, 1], colors sRGB in [0, 1].
struct RampStop {
    float position = 0.0f;
    std::array<float, 3> rgb{};
    bool operator==(const RampStop&) const = default;
};
// A Color ramp node's stops in stored order (its Stops count of them; empty for other kinds).
std::vector<RampStop> ramp_stops(const Node& ramp);
// Sets count and stops (clamped to the parameter ranges, at most ramp_node::kMaxStops; fewer
// than 2 are padded with white at 1). Returns false if `id` is not a Color ramp.
bool set_ramp_stops(Graph& graph, int id, const std::vector<RampStop>& stops);
// The palette's key colors as evenly spaced stops.
std::vector<RampStop> palette_ramp(DisplayPalette palette);
// Color at t of stops sorted by position (stable): the first color up to the first stop, the
// last from the last stop on, linear in between. Non-finite t = 0.
std::array<float, 3> sample_ramp(const std::vector<RampStop>& sorted, float t);

enum class DisplayOpCode : std::uint32_t {
    Const = 0, // (k0, k1, k2, k3)
    Field = 1, // field m0 (DisplayField) as (v, v, v, 1)
    Ramp = 2,  // ramp of r[a].x: t = clamp((v − k0) / (k1 − k0)), stops m0 .. m0 + m1 − 1
    Mix = 3,   // blend r[b] into r[a] by clamp(r[c].x, 0, 1), mode m0 (mix_node::ModeChoice)
};

// 48 bytes, the GPU's std430 struct of 12 scalars (as RuleOp).
struct DisplayOp {
    std::uint32_t op = 0;
    std::uint32_t a = 0, b = 0, c = 0;
    std::uint32_t m0 = 0, m1 = 0;
    float k[6] = {};
    bool operator==(const DisplayOp&) const = default;
};
static_assert(sizeof(DisplayOp) == 48);

// Limits of the GPU program buffer (workbench_view.frag).
constexpr std::size_t kMaxDisplayOps = 32;
constexpr std::size_t kMaxDisplayStops = 64;
constexpr std::uint32_t kNoDisplayRegister = 0xffffffffu;

struct DisplayProgram {
    std::vector<DisplayOp> ops;
    // Ramp stops of every Ramp op, each ramp's sorted by position: position, r, g, b.
    std::vector<std::array<float, 4>> stops;
    std::uint32_t output = kNoDisplayRegister; // register holding the color
    bool operator==(const DisplayProgram&) const = default;
};

struct DisplayCompile {
    bool linked = false; // the View node's input is linked; otherwise the panel display applies
    DisplayProgram program;
    std::string error; // non-empty: the graph does not fit the GPU limits (program is empty)
};

// Compiles a display graph at `generation`. Not linked (and no error) for a graph without a
// View node or with its input unlinked, or for a simulation graph.
DisplayCompile compile_display(const Graph& graph, std::uint64_t generation);

// sRGB color (RGBA, each in [0, 1]) of a cell with the given field values.
std::array<float, 4> run_display(const DisplayProgram& program, float density, float neighborhood, float growth);

// The program as the GPU buffer holds it: count, output, stop count, 0, then kMaxDisplayOps
// ops of 12 words (unused ones zero), then kMaxDisplayStops stops of 4 floats.
constexpr std::size_t kDisplayBufferWords = 4 + 12 * kMaxDisplayOps + 4 * kMaxDisplayStops;
std::vector<std::uint32_t> pack_display(const DisplayProgram& program);

enum class DisplayPreset {
    Panel,         // a View node alone: the Simulation panel's display
    WiredPanel,    // the panel's field, palette and range as Fields -> Color ramp -> View
    GrowthOverlay, // example: density ramp, growth ramp screened over it, pulsing with time
};
// `field` and `range` are the panel display WiredPanel reproduces; the others ignore them.
Graph make_display_preset(DisplayPreset preset, DisplayField field = DisplayField::Density,
                          const DisplayRange& range = {});
// Adds Fields and a Color ramp equivalent to the panel display (field, palette, range) and
// links them into the View node: the same picture, now editable. Returns false, changing
// nothing, if the View input is already linked or the graph is not a display graph.
bool wire_panel_display(Graph& graph, DisplayField field, const DisplayRange& range);

} // namespace lenia
