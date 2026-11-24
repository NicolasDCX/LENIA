# Node graph: per-cell rule, kernel and environment over time

The **Node graph** panel (bottom dock of `lenia_workbench`; *View* menu reopens it, *Help → Node
graph quick guide* summarizes it) defines the **per-cell update rule** and changes the kernel
and a spatial environment while the simulation runs. The graph is evaluated **once per
generation**, right before that generation's Vulkan dispatch, so rule constants, kernel and
environment can change between consecutive generations inside one frame; the rule itself then
runs for every cell on the GPU.

## Nodes

Port colors: orange **Value** (one number per generation), teal **Kernel**, green
**Environment**, pink **Cell** (one number per cell). Links join equal types; a Value output
may also feed a Cell input, which then gets the same value in every cell.

| Node        | Inputs                          | Outputs            | What it does |
|-------------|---------------------------------|--------------------|--------------|
| Time        | –                               | Value              | `Start + Speed × generation` |
| Oscillator  | Time, Amplitude, Offset         | Value              | `Offset + Amplitude × wave(Time / Period + Phase)`; sine, triangle, square, saw. Unlinked Time = the generation |
| Constant    | –                               | Value              | A fixed value |
| Kernel      | Radius Δ, Peak shift, Sharpness | Kernel             | Warps the Kernel panel's ring shape (below) |
| Environment | Strength, Offset, Angle         | Environment, Field | Spatial growth field (below); Field = `Strength × E(x, y)` per cell |
| Simulation  | Kernel, Environment, Next state | –                  | What the GPU uses (below) |
| Density A, Neighborhood U, Growth, Add, Multiply, Clamp, Compare, Select, Life transition | | Cell | Cell rule nodes, see *Cell rules* |

A linked **Value** input **adds** to the parameter of the same name (shown as "(+ link)" in the
inspector), so the parameter is the base value and the link is the modulation. A linked
**Cell** input **replaces** its parameter ("(linked: unused)"); unlinked, the parameter is the
value in every cell.

## Cell rules

The **Simulation** node's *Next state* input is the next state of each cell. Whatever is wired
into it is compiled each generation (`lenia::compile_rule`, `include/lenia/rule.hpp`) into a
straight-line program of at most 64 operations (every node output, constant and unlinked
parameter counts one) and run for every cell by the Vulkan step (`shaders/lenia_rule.glsl`,
included by `workbench_step.comp`). The CPU reference `lenia::step_rule` / `run_rule` runs the
same programs op for op.

| Node            | Inputs (unlinked default)                 | Output |
|-----------------|-------------------------------------------|--------|
| Density A       | –                                         | the cell's current state |
| Neighborhood U  | –                                         | `U = K ∗ A` with the **active kernel**: the one Simulation's Kernel input gives (warped by a Kernel node, or the Kernel panel's static kernel). With normalization *None* it is a raw weighted count |
| Growth G(U)     | U (= Neighborhood), μ, σ (parameters)     | G = Lenia growth of U (Gaussian `2·exp(−z²/2) − 1` or polynomial `2·max(0, 1 − z²/9)⁴ − 1`, `z = (U − μ)/σ`); dt (Value) = the step size. *Source = Simulation panel* follows the panel's function, μ, σ, dt (what Restore Orbium sets); *This node* uses the node's own |
| Add, Multiply   | A, B                                      | `A + B`, `A × B` |
| Clamp           | X, Low (0), High (1)                      | `min(max(X, Low), High)` |
| Compare         | A, B; test `> ≥ < ≤ = ≠`                  | 1 where the test holds, else 0 (exact float comparison) |
| Select          | Condition, If true (1), If false (0)      | If true where Condition ≥ 0.5, else If false |
| Life transition | State (= Density), Neighbors (= Neighborhood); Birth mask, Survive mask | exact life-like rule: alive = State ≥ 0.5, `n = floor(Neighbors + 0.5)`; next = bit n of Survive (alive) or Birth (dead), 0 outside 0..8. A mask adds 2ⁿ per count n: Conway's B3/S23 is Birth 8, Survive 12; the node shows the rule name |

Any other Value node (Time, Oscillator, Constant, Growth's dt) linked into a cell input is a
per-generation constant, so a rule can be modulated over time. Results that are not finite
(say σ linked to 0) become 0 on CPU and GPU alike, so a NaN cannot spread through the world.

With a rule linked, the Simulation node's **Environment input is not used**: the rule adds the
environment where it wants it, through an Environment node's *Field* output. The Simulation
panel's growth controls only reach the rule through Growth nodes set to *Source = Simulation
panel*.

**Next state unlinked** = the built-in update `clamp(A + dt × (G(U) + Strength × E), 0, 1)`
with the panel's growth and the Environment linked into Simulation, exactly as before rule
nodes existed. That is how experiments saved before rule nodes open. Select the Simulation
node and press *Wire the built-in Lenia rule as nodes* (also in its right-click menu;
`lenia::wire_lenia_rule`) to turn it into the equivalent nodes: Neighborhood → Growth (panel)
→ Add Field → Multiply by dt → Add Density → Clamp → Next state, with an Environment linked into
Simulation moved to the rule's Field. The result is bit for bit the same on the CPU
(`tests/test_rule.cpp`).

### Default graph and examples

- The workbench starts with **Lenia rule, idle**: the Time/Oscillator/Kernel/Environment nodes
  of before, nothing modulated, and the Lenia update wired as above, so Orbium runs exactly as
  with the built-in update. *Breathing kernel* and *Drifting environment* are wired the same
  way.
- **Game of Life (Life node)** and **Game of Life (Compare / Select)** load Conway's B3/S23:
  Density and Neighborhood into a Life transition, or the same rule from nodes as
  `Select(A, (U = 2) + (U = 3), U = 3)`. Loading either (or the *Game of Life (glider)* button
  in the Simulation panel's Presets) replaces the whole setup with `lenia::make_life_project`:
  radius 1 kernel whose paint is the **raw 8-neighbor Moore kernel** (weights 1 around a 0,
  normalization *None*, so U is the live-neighbor count, exactly), a **glider** in the middle
  of a 128 × 128 world, grayscale density and a 0–8 neighborhood range. The glider moves one
  cell right and down every 4 generations. *Restore Orbium* returns to Lenia (Orbium, the
  default graph, a 256 world and the Lenia display ranges).

### Display

How the fields are colored is the separate **display graph** (`display_graph.md`); with its
View node unlinked (the default) the Simulation panel's *View* settings apply. The *Neighborhood* field shows U of the active kernel. The *Growth* field shows the rule's
first Growth node output (without the environment term), or `next − current` when the rule has
no Growth node (the Game of Life: +1 births, −1 deaths); without a rule it is the built-in
G(U) including the environment term, as before.

## Kernel warp (`lenia::KernelWarp`, `kernel_shape.hpp`)

The shape itself changes, not just its brightness, so the effect survives normalization:

- **Radius Δ**: effective radius `Re = R × (1 + Δ)`; the kernel grid grows or shrinks with it
  (capped at the workbench's 64-cell limit).
- **Peak shift**: each ring's coordinate `t → t^(2^shift)`; positive moves ring peaks outward.
- **Sharpness**: core value `c → c^(2^sharp)`; positive narrows rings, negative widens them.

All zero is exactly the static kernel, including painted cells (paint is rescaled with the
radius when warped). The Kernel node draws the warped radial profile over the static one
(dashed).

## Environment (`lenia::Environment`, `environment.hpp`)

The built-in update becomes `A' = clamp(A + dt × (G(U) + Strength × E(x, y)), 0, 1)` with
`E ∈ [−1, 1]`, and the wired Lenia rule computes the same through the *Field* output: positive
regions favor growth, negative regions suppress it. Patterns, around the world center, with `u`
the axis along *Angle* and *Offset* in cells:

| Pattern  | E(x, y) |
|----------|---------|
| Uniform  | 1 (global bias) |
| Spot     | `cos(π · min(d / Scale, 1))`, center moved by Offset along Angle |
| Gradient | `clamp((u − Offset) / Scale, −1, 1)` |
| Stripes  | `sin(2π (u − Offset) / Scale)` |
| Ripples  | `cos(2π (d − Offset) / Scale)`, expanding as Offset grows |

Link **Time → Offset** to make it drift, an **Oscillator → Strength** to make it pulse. The
Environment node previews the strength-signed field over the world (red boosts, blue
suppresses); the *Growth G(U)* view includes the term. `workbench_step.comp` mirrors the CPU
formulas (`lenia_math.glsl`).

## Editing

- Drag from an output port to a compatible input (same color) to link; a new link replaces the
  old one. Links that would make a node depend on itself are refused (status bar says why).
- Drag a linked input away to unlink it, or right-click it / the link → *Unlink*.
- Right-click empty space → *Add node* / *Add cell rule node*; or use the toolbar menus *Add
  value / modulation* and *Add cell rule node*. Del / Backspace deletes the selection; Home
  frames all; wheel zooms; drag empty space (or middle drag) pans.
- Select a node to edit its parameters in the inspector on the right.
- *Example → Load*: **Lenia rule, idle** (default: nothing modulated, Orbium runs as with the
  built-in update), **breathing kernel**, **drifting environment**, each neutral at generation
  0; **Game of Life** (Life node, or Compare / Select), which load the whole Life setup.
- A rule over 64 operations is refused: the Simulation node says *Rule: too large, not
  stepping*, the status bar says why, and the world does not advance until it is simplified.

## Determinism

Graph values are a pure function of the generation number. Pause, *Step ×N* (N single
generations, each with its own kernel/environment), several generations per frame, and
**Reset (R)**, which restarts at generation 0, all replay identically.
`tests/test_node_graph.cpp` checks this on the CPU reference (`lenia::step_graph`), that the
idle graph runs Orbium bit-for-bit unchanged, and that warps change the kernel's profile.

`tests/test_rule.cpp` checks the rules on the CPU: the glider equals its start shifted by
(1, 1) after 4 generations (and wraps the 16 × 16 torus in 64) for both Life presets, the
Compare / Select wiring equals the Life node on random soup, blinker and HighLife masks, the
Life experiment's kernel is the Moore kernel and roundtrips, the wired Lenia rule and a
migrated pre-rule graph (with a drifting environment) equal `step_lenia` bit for bit, a
rewired rule matches its formula cell by cell, oscillators modulate cell inputs per
generation, and oversized rules are refused. `tests/test_vulkan_parity.cpp` runs rules through
the GPU interpreter (`VulkanStepper::step_rule`, `shaders/rule_step.comp`, which includes the
same `lenia_rule.glsl` as the workbench): a rewired rule using every op (growth with its own
μ / σ, the environment Field, Life, Compare, Select, Clamp, Value → Cell) within 1e-4 of the
CPU over 1 and 3 generations, both Life presets **exactly** over 40 generations, the glider
after 4 GPU generations, and the default Lenia rule on a settled Orbium.

## Saving

*File → Save experiment* stores the graph completely (node ids, positions, parameters, links,
rule nodes included) together with the kernel, growth, world seed and display; reopening
restarts at generation 0 and replays identically. See `projects.md`. *File → Save snapshot*
also stores the running world and its generation, so the graph continues exactly where it was
(`snapshots.md`).
