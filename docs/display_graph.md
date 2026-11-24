# Display graph: how the viewport colors each cell

The **Display graph** panel of `lenia_workbench` (tabbed with the node graph at the bottom; *View*
menu reopens it, *Help → Display graph quick guide* summarizes it) decides the color of every
cell in the viewport. It is a second node graph, separate from the simulation graph
(`node_graph.md`): it reads the simulation's fields and never changes them, so rules, kernels and
modulation behave exactly as without it.

The graph is compiled on the CPU (`lenia::compile_display`, `include/lenia/display_graph.hpp`)
into a straight-line program of at most 32 operations and 64 ramp stops, uploaded when it
changes, and run **per pixel by the fragment shader** (`shaders/workbench_view.frag`) every
frame, whether the simulation is running or paused. Edits show immediately while paused. The
CPU reference `lenia::run_display` runs the same programs op for op.

## Nodes

Port colors as in the node graph, plus yellow **Color**. Pink ports carry one number per cell;
an orange Value output (Time, Oscillator, Constant) may feed a pink input and then gives the same
value in every cell, fixed per generation.

| Node       | Inputs (unlinked)                         | Outputs | What it does |
|------------|-------------------------------------------|---------|--------------|
| Fields     | –                                         | Density A, Neighborhood U, Growth, Nutrient N | The cell's fields, exactly what the panel's *Field* choice shows (Growth: the rule's first Growth node, or next − current; Nutrient: the second state field, `state_fields.md`) |
| Color ramp | Value (= density); Low, High; stops       | Color   | `t = clamp((Value − Low) / (High − Low), 0, 1)` (0 when High = Low), then the color at `t` between the stops, linear in sRGB |
| Color      | –; Red, Green, Blue                       | Color   | A fixed sRGB color |
| Mix        | A, B (black); Factor (parameter); Mode    | Color   | `f = clamp(Factor, 0, 1)`; **Mix** `A + (B − A)·f`, **Add** `A + B·f`, **Multiply** `A·(1 + (B − 1)·f)`, **Screen** `1 − (1 − A)(1 − B·f)`; each channel clamped to [0, 1] |
| View       | Color (the panel display)                 | –       | What the viewport shows |
| Time, Oscillator, Constant | as in the node graph      | Value   | Per-generation values, e.g. into a Mix factor |

Unlike a Value input in the simulation graph, a linked pink input **replaces** its parameter
(as the cell rule nodes do). NaN field values count as 0 wherever they are clamped.

### Color ramps

A ramp has 2 to 10 stops, each a position in [0, 1] and a color. Before the first stop the first
color holds, after the last the last; two stops at the same position make a hard edge. Stops
may be stored in any order: they are sorted by position (stably) when compiled.

Select a Color ramp to edit it in the inspector: **drag** a stop's handle to move it,
**double-click** a handle to choose its color, **double-click** the bar to add a stop with the
color already there, **right-click** a handle to remove it (two always remain). *Set stops from
palette…* copies a built-in palette's key colors as evenly spaced stops (all of them fit:
Viridis and Inferno have 10), and *Reverse* mirrors the stops.

### The View node and the panel display

**View unlinked** = the Simulation panel's *View* settings (field, palette, low, high) through
the 256-entry palette tables, exactly as before display graphs existed. This is the default, and
how experiments saved before format version 3 open. While View is linked, the panel's field and
palette controls are disabled and a note says why.

*Wire the panel display as nodes* (View node's inspector or right-click menu;
`lenia::wire_panel_display`) adds Fields and a Color ramp holding the panel's current field,
palette, low and high, linked into View: the same picture, now editable. It matches the panel
on every palette table entry; between entries the ramp is continuous while the panel steps in
1/255 increments (`tests/test_display_graph.cpp`).

A display graph over the GPU limits is not used: the view falls back to the panel display, the
View node says *Too large*, and the status bar says why.

## Examples

*Example → Load* in the Display graph toolbar replaces the display graph only:

- **Panel display** – a lone View node (the default).
- **Panel display as nodes** – the panel's current field and palette, wired as above.
- **Density + pulsing growth overlay** – density through a deep-sea ramp; growth through a
  blue–black–orange ramp over [−1, 1] (blue shrinking, black steady, orange growing), screened
  over it with a factor an Oscillator swings between 0.25 and 0.75 every 240 generations. Where
  growth is 0 the plain density ramp shows.
- **Density over nutrient** (`DisplayPreset::DensityNutrient`, what the nutrient trail demo
  uses) – the nutrient through a soil ramp (brown where eaten, green where full), with the
  density through a black–blue–white ramp screened over it at factor 1.

## Determinism and saving

The program is a pure function of the display graph and the generation, so pausing, stepping
and Reset (R) replay the picture exactly. *File → Save experiment* stores the display graph in
the `[display_graph]` section (format version 3, `projects.md`); reopening restores it bit for
bit.
