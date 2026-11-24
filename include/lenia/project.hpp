#pragma once

#include "lenia/convolve.hpp"
#include "lenia/display_graph.hpp"
#include "lenia/kernel_shape.hpp"
#include "lenia/node_graph.hpp"
#include "lenia/pattern.hpp"
#include "lenia/step.hpp"

#include <array>
#include <istream>
#include <optional>
#include <string>

namespace lenia {

// A saved workbench experiment: everything needed to restart it at generation 0 and replay it
// exactly (docs/projects.md). It stores the setup, not the running state: the world is rebuilt
// from the seed settings, and the node graph is a pure function of the generation.

// Format version written by write_project; parse_project reads versions 1..kProjectVersion.
// Version 2 adds the cell rule nodes, the Environment's Field output and the Simulation node's
// Next state input; a version 1 file has none of them and runs the built-in Lenia update.
// Version 3 adds the [display_graph] section; older files get a lone View node, so they show
// the panel display ([display]) exactly as before.
constexpr int kProjectVersion = 3;

// Workbench seed choices, in the order of its Seed menu.
enum class SeedKind { Random, Pattern, Impulse, Empty };
// Display fields, palettes and ranges (DisplayField, DisplayPalette, DisplayRange) are in
// display_graph.hpp.

struct Project {
    // Base kernel: ring shape, the additive paint layer ((2R + 1)^2 raw weights, or empty for
    // none) and the normalization of shape + paint.
    KernelShape kernel;
    KernelNormalization normalization = KernelNormalization::Sum;
    Grid paint;

    GrowthParams growth;

    // World and seed. `pattern` keeps only its name, source, license and cells; its kernel and
    // growth fields are not stored (the project's own kernel and growth are what runs).
    int world_size = 256;
    SeedKind seed = SeedKind::Pattern;
    int random_seed = 1;
    std::optional<Pattern> pattern;

    // Display: the field shown and each field's palette and range.
    DisplayField field = DisplayField::Density;
    std::array<DisplayRange, kDisplayFieldCount> display{};

    // Modulation graph with node positions; the environment is its Environment node.
    Graph graph;

    // Display graph (Display domain). Its View node unlinked = the panel display above.
    Graph display_graph = make_display_preset(DisplayPreset::Panel);
};

// The project as text in the current format version. Floats are written in their shortest
// exact form, so parse_project(write_project(p)) restores every value bit for bit.
std::string write_project(const Project& project);

// Parses a project. Strict: the "lenia-project <version>" header comes first, every section
// and key the version defines must be present exactly once, and unknown keys, malformed or
// non-finite numbers, out-of-range values, grids of the wrong size and invalid graphs (see
// Graph::assign) are rejected. Throws std::invalid_argument with the line number.
Project parse_project(std::istream& in);

// Conway's Game of Life as an experiment: the raw Moore kernel (radius 1, painted 8-neighbor
// ring, normalization none), a glider seeded in the middle of a 128 x 128 world, a grayscale
// display and the given Life graph preset (GameOfLife or GameOfLifeCompare).
Project make_life_project(GraphPreset preset = GraphPreset::GameOfLife);

} // namespace lenia
