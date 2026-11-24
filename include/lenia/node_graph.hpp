#pragma once

#include "lenia/convolve.hpp"
#include "lenia/environment.hpp"
#include "lenia/kernel_shape.hpp"
#include "lenia/step.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace lenia {

// Modulation graph: value nodes (Time, Oscillator, Constant) drive the inputs of a Kernel and
// an Environment node, which feed the single Output node. The graph is evaluated once per
// generation, as a pure function of the generation number, so pausing, single steps, batched
// steps and resets replay identically (docs/node_graph.md).
//
// Cell rule nodes (Density ... Life) compute a per-cell value; wired into the Output node's
// Next state input they replace the built-in Lenia update with the graph's own rule, which
// compile_rule (rule.hpp) turns into a small program the GPU runs for every cell. With Next
// state unlinked the built-in update runs, exactly as before rule nodes existed.
//
// The same Graph class also holds the display graph (display_graph.hpp): a graph in the Display
// domain whose nodes (Fields, Color ramp, Color, Mix, View) color the viewport. Each domain has
// its own node kinds and its own sink (Simulation's Output, the display's View); the Value
// nodes (Time, Oscillator, Constant) exist in both.

enum class NodeKind {
    Time, Oscillator, Constant, Kernel, Environment, Output,
    // Cell rule nodes, appended so saved kinds keep their meaning.
    Density, Neighborhood, Growth, Add, Multiply, Clamp, Compare, Select, Life,
    // Display nodes (display graph only).
    Fields, ColorRamp, Color, Mix, View,
};
enum class PortType { Value, Kernel, Environment, Cell, Color };
enum class GraphDomain { Simulation, Display };

// Whether a node kind belongs in graphs of the domain, and the domain's single sink node.
bool in_domain(NodeKind kind, GraphDomain domain);
NodeKind sink_kind(GraphDomain domain);
bool is_sink(NodeKind kind);
enum class Waveform { Sine, Triangle, Square, Saw };
enum class CompareOp { Greater, GreaterEqual, Less, LessEqual, Equal, NotEqual };

// Outputs link into inputs of the same type; a Value output also links into a Cell input,
// which then gets the same value in every cell.
bool can_link(PortType from, PortType to);

struct ParamSpec {
    std::string name;
    float min, max, def, step;
    int decimals;
    std::vector<std::string> choices; // non-empty: an enum, the value is the choice index
};

struct PortSpec {
    std::string name;
    PortType type;
    // Value inputs: the parameter the input adds to when connected and stands in for when not.
    // Cell inputs: the parameter used while unlinked; a link replaces it.
    // -1 for an input without one: Oscillator's Time defaults to the generation, Growth's U and
    // Life's Neighbors to the cell's neighborhood, Life's State to the cell's density.
    int param = -1;
};

struct NodeSpec {
    NodeKind kind;
    std::string name;
    std::string help;
    std::vector<PortSpec> inputs;
    std::vector<PortSpec> outputs;
    std::vector<ParamSpec> params;
};

const NodeSpec& node_spec(NodeKind kind);
const std::vector<NodeKind>& node_kinds(); // every kind, in enum order
const std::vector<NodeKind>& node_kinds(GraphDomain domain);

// Parameter and port indices, matching node_spec().
namespace time_node { enum Param { Speed, Start }; }
namespace oscillator_node {
enum Param { Wave, Period, Phase, Amplitude, Offset };
enum Input { TimeIn, AmplitudeIn, OffsetIn };
}
namespace constant_node { enum Param { Value }; }
namespace kernel_node { enum Param { RadiusDelta, PeakShift, Sharpness }; } // inputs share the indices
namespace environment_node {
enum Param { Pattern, Strength, Scale, Angle, Offset };
enum Input { StrengthIn, OffsetIn, AngleIn };
enum Output { EnvironmentOut, FieldOut }; // FieldOut: Strength × E(x, y) per cell
}
namespace output_node { enum Input { KernelIn, EnvironmentIn, NextIn }; }
namespace growth_node {
enum Param { Source, Function, Mu, Sigma, Dt };
enum Input { UIn, MuIn, SigmaIn };
enum Output { GOut, DtOut };
enum SourceChoice { Panel, Own }; // Source: the Simulation panel's growth, or this node's
}
namespace add_node { enum Param { A, B }; } // inputs share the indices
namespace multiply_node { enum Param { A, B }; }
namespace clamp_node { enum Param { X, Low, High }; }
namespace compare_node {
enum Param { Op, A, B };
enum Input { AIn, BIn };
}
namespace select_node { enum Param { Condition, IfTrue, IfFalse }; }
namespace life_node {
enum Param { Birth, Survive }; // bit n set: n neighbors give birth / let a live cell survive
enum Input { StateIn, NeighborsIn };
}
// Display nodes.
namespace fields_node { enum Output { DensityOut, NeighborhoodOut, GrowthOut }; }
namespace ramp_node {
// Low and High map the input to t in [0, 1]; Stops (2..kMaxStops) stops follow, each as
// position, red, green, blue (sRGB in [0, 1]). Stops past the count are stored but unused.
enum Param { Low, High, Stops, FirstStop };
enum Input { ValueIn }; // unlinked: the density
enum Component { Position, Red, Green, Blue };
constexpr int kMaxStops = 10;
constexpr int stop_param(int stop, int component) { return FirstStop + 4 * stop + component; }
}
namespace color_node { enum Param { Red, Green, Blue }; }
namespace mix_node {
enum Param { Mode, Factor };
enum Input { AIn, BIn, FactorIn }; // unlinked A, B: black
enum ModeChoice { Mix, Add, Multiply, Screen };
}
namespace view_node { enum Input { ColorIn }; } // unlinked: the Simulation panel's display

struct Node {
    int id = 0;
    NodeKind kind = NodeKind::Constant;
    std::vector<float> params;
    float x = 0.0f; // canvas position of the top-left corner
    float y = 0.0f;
};

struct Link {
    int from_node, from_port; // an output
    int to_node, to_port;     // an input
    bool operator==(const Link&) const = default;
};

enum class ConnectResult { Ok, BadPort, TypeMismatch, Cycle };
const char* describe(ConnectResult r);

class Graph {
public:
    Graph() = default; // a simulation graph
    explicit Graph(GraphDomain domain) : domain_(domain) {}

    GraphDomain domain() const { return domain_; }

    // Adds a node with the spec's default parameters and returns its id, or -1 if the kind
    // does not belong in this graph's domain. Only one sink (Output / View) node can exist:
    // adding another returns the existing one.
    int add_node(NodeKind kind, float x = 0.0f, float y = 0.0f);
    // Removes the node and its links. The sink node cannot be removed.
    bool remove_node(int id);

    const Node* node(int id) const;
    Node* node(int id);
    const std::vector<Node>& nodes() const { return nodes_; }
    const std::vector<Link>& links() const { return links_; }
    // The sink node: Simulation's Output, or the display graph's View.
    int output_node() const;

    // Links an output to an input, replacing the input's existing link. Rejects links between
    // port types can_link refuses and links that would make a node depend on itself.
    ConnectResult connect(int from_node, int from_port, int to_node, int to_port);
    // Removes the link into the input, if any.
    bool disconnect(int to_node, int to_port);
    const Link* link_into(int to_node, int to_port) const;

    // Clamped to the spec's range (enums to a valid choice).
    void set_param(int id, int index, float value);
    void move_node(int id, float x, float y);

    // Replaces the whole graph with saved nodes and links (ids and order kept), after checking
    // all of it: positive unique ids, kinds of this graph's domain, exactly one sink node,
    // finite positions, the spec's parameter count with values in range (enums integral),
    // and links that connect() accepts
    // with at most one link per input. Returns an empty string on success; otherwise what is
    // wrong, and the graph is unchanged.
    std::string assign(const std::vector<Node>& nodes, const std::vector<Link>& links);

private:
    bool depends_on(int node, int upstream) const;

    GraphDomain domain_ = GraphDomain::Simulation;
    std::vector<Node> nodes_;
    std::vector<Link> links_;
    int next_id_ = 1;
};

// What the Output node receives at one generation.
struct GraphFrame {
    bool kernel_connected = false;
    KernelWarp kernel;
    // Only applied by the built-in update; a rule adds the environment through the
    // Environment node's Field output instead.
    bool environment_connected = false;
    Environment environment;
    bool rule_connected = false; // Next state is linked: compile_rule (rule.hpp) gives the update
};

// Empty (nothing connected) for a display graph.
GraphFrame evaluate(const Graph& graph, std::uint64_t generation);

// Value of a Value output (0 for other ports), and of a Value input as the node sees it
// (link or parameter); for display. `panel` is the Simulation panel's growth, which a Growth
// node set to Source = panel reports on its dt output.
double output_value(const Graph& graph, int node, int port, std::uint64_t generation,
                    const GrowthParams& panel = {});
double input_value(const Graph& graph, int node, int port, std::uint64_t generation,
                   const GrowthParams& panel = {});
KernelWarp node_kernel(const Graph& graph, int node, std::uint64_t generation);
Environment node_environment(const Graph& graph, int node, std::uint64_t generation);
// Function, mu, sigma and dt a Growth node uses while its mu and sigma inputs are unlinked.
GrowthParams node_growth(const Graph& graph, int node, const GrowthParams& panel);

double waveform(Waveform wave, double x); // period 1, in [-1, 1], 0 at x = 0 except Square/Saw

// "B3/S23" for a Life node's masks.
std::string life_rule_name(int birth_mask, int survive_mask);

// Built-in graphs. The Lenia ones start at neutral values at generation 0 and wire the Lenia
// update visibly (wire_lenia_rule), so they run Orbium exactly as the built-in update does.
enum class GraphPreset {
    Idle,                // every node present, nothing modulated: Orbium runs as without a graph
    BreathingKernel,     // two oscillators warp the kernel radius and ring peak
    DriftingEnvironment, // slow stripes drift across the world, strength pulsing
    GameOfLife,          // Conway's B3/S23 with a Life node; needs the raw Moore kernel (rule.hpp)
    GameOfLifeCompare,   // the same rule from Compare, Add and Select nodes
};
Graph make_preset(GraphPreset preset);
// Whether the preset is a Game of Life rule, which needs the Moore kernel rather than a Lenia one.
bool is_life_preset(GraphPreset preset);

// Adds Density, Neighborhood, Growth (Source = panel), Add, Multiply and Clamp nodes computing
// clamp(A + dt × (G(U) + field), 0, 1) and links them into Next state: the built-in update made
// visible and editable. An Environment linked into the Output node is moved to the rule (its
// Field output feeds the sum), so the result is unchanged. Nodes are placed in a row starting
// at (x, y); the Output node moves to its end. Returns false, changing nothing, if Next state is
// already linked or there is no Output node.
bool wire_lenia_rule(Graph& graph, float x, float y);
// The same, placed below the existing nodes; for migrating graphs saved before rule nodes.
bool wire_lenia_rule(Graph& graph);

// CPU reference of one workbench generation: the kernel is make_kernel(shape, warp) normalized
// to Sum = 1 (make_kernel(shape) when the kernel input is unconnected).
Grid graph_kernel(const Graph& graph, const KernelShape& shape, std::uint64_t generation);
Grid step_graph(const Grid& world, const Graph& graph, const KernelShape& shape, const GrowthParams& growth,
                std::uint64_t generation);
// The same with an explicit kernel (the Game of Life's raw Moore kernel, say): the graph's rule
// (step_rule) when Next state is linked, else step_lenia with the graph's environment. Throws
// std::invalid_argument if the rule does not compile.
Grid step_graph(const Grid& world, const Graph& graph, const Grid& kernel, const GrowthParams& growth,
                std::uint64_t generation);

} // namespace lenia
