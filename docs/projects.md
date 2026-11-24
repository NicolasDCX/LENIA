# Experiment files (`.leniaexp`)

*File → Open experiment… (Ctrl+O)*, *Save experiment (Ctrl+S)* and *Save experiment as…
(Ctrl+Shift+S)* in `lenia_workbench` store a whole experiment so it can be reopened and
replayed exactly. The format is implemented in `lenia::Project` (`include/lenia/project.hpp`,
`src/project.cpp`) and has no Qt dependency; `tests/test_project.cpp` covers it.

## What is stored

| Part      | Contents |
|-----------|----------|
| Kernel    | Radius, core, signed ring weights, normalization, and the painted layer cell by cell |
| Growth    | Function (Gaussian / polynomial), μ, σ, dt |
| World     | Size, seed choice (random square, loaded pattern, impulse, empty), random seed |
| Nutrient  | How the second state field starts (uniform, random, gradient) and its level (`state_fields.md`) |
| Pattern   | The loaded pattern's name, source, license and cells, if one is loaded (even when the seed is not "Loaded pattern", so switching to it later still works) |
| Display   | The field shown, and each field's palette, low and high |
| Graph     | Every node (id, kind, canvas position, all parameters) and every link, in order, cell rule nodes included. The environment is the Environment node's parameters and links; the update rule is whatever is linked into the Simulation node's Next state |
| Display graph | The same for the display graph (`display_graph.md`): Fields, Color ramp (stops included), Color, Mix, View and Value nodes and their links |

Floats are written in their shortest exact form, so every value reads back bit for bit.

**Not stored:** the running world. Opening an experiment rebuilds the world from its seed, the
same way *Reset (R)* does, and starts **paused at generation 0** (the status bar says so; Space
runs it). Because the node graph is a pure function of the generation (`node_graph.md`), the run
then replays exactly. Cells painted into the world by hand after the last reset are not part of
the experiment. Also not stored: UI-only settings (brush sizes, Step ×N, generations per frame,
dock layout). To keep a running world and continue it later, save a **snapshot** instead
(`snapshots.md`): it embeds this format unchanged, plus the density and the generation.

## Opening is all or nothing

The file is read and checked completely before anything in the workbench changes. If any check
fails, a message explains which line and why, and the current experiment is untouched. Rejected:

- a missing or wrong `lenia-project <version>` header, or a version newer than the build reads;
- in a version 1 file, rule node kinds, the Environment's Field output or the Simulation's
  Next state input (none existed then); before version 3, a `[display_graph]` section; before
  version 4, a `[nutrient]` section, the `nutrient` node kind or display key, and links into the
  Simulation's Next nutrient or out of Fields' Nutrient; from version 4, a missing `[nutrient]`;
- node kinds in the wrong graph: display nodes in `[graph]`, simulation nodes (Kernel,
  Environment, Simulation, cell rule nodes) in `[display_graph]`; a display graph without
  exactly one View node;
- missing, duplicate or unknown sections and keys, and extra values on a line;
- malformed, `nan` / `inf` or out-of-range numbers (σ and dt must be positive, radius ≥ 1, …);
- unknown names (core, growth function, seed, palette, node kind, …);
- paint that is not `(2R + 1)²` cells, paint values outside [−2, 2], grids with missing or
  short rows;
- invalid graphs (`Graph::assign`): not exactly one Simulation node, duplicate ids, the wrong
  number of parameters, parameters outside their range or fractional enum values and Life
  masks, links to missing ports, between port types (other than Value → Cell), into an input
  that already has one, or forming a cycle;
- values the workbench's controls cannot hold exactly: kernel radius above 64, μ / σ / dt /
  display ranges outside their spin boxes, world sizes other than 128, 256, 512 or 1024, random
  seeds above 2³⁰. (Values with more decimals than a spin box shows are rounded to it; files the
  workbench writes already are.)

## Saving is atomic

Saving goes through `QSaveFile`: the text is written to a temporary file next to the target and
renamed over it only once complete, so a failed save (disk full, permissions) leaves an existing
file intact. Before writing, the workbench parses its own output again and refuses to save
anything it could not reopen.

## Format, version 5

Plain UTF-8 text, one `key values` line each; blank lines and lines starting with `#` are
ignored. There are no end-of-line comments: a `#` inside a line is data (a pattern source URL
may contain one), so the parser rejects `mu 0.15 # note` as an extra value. Text values (pattern
name, source, license) are the rest of the line, trimmed. Sections may come in any order;
`[pattern]` is optional, the others are required (`[display_graph]` from version 3 on,
`[nutrient]` from version 4 on).

```
lenia-project 5

[kernel]
radius 13
# exponential | polynomial | step | gaussian
core polynomial
# one signed weight per ring, innermost first
rings 1
# none | sum | positive | abs_sum
normalization sum
# "none", or "paint W H" followed by H rows of W numbers, top row first
paint none

[growth]
# gaussian | polynomial
function polynomial
mu 0.15
sigma 0.015
dt 0.1

[world]
size 256
# random | pattern | impulse | empty
seed pattern
random_seed 1

[nutrient]
# uniform | random | gradient
start uniform
# in [0, 1]
level 1

[pattern]
name Orbium
source ...
license ...
# followed by 20 rows of 20 numbers
cells 20 20

[display]
# density | neighborhood | growth | nutrient
field density
# <field> <palette> <low> <high>; grayscale | viridis | inferno | redblue | icefire
density viridis 0 1
neighborhood inferno 0 0.3
growth redblue -1 1
nutrient inferno 0 1

[display_graph]
# same node / link lines as [graph]; a lone "node 1 view 560 40" shows the panel display
node 1 fields -500 40
node 2 ramp -240 40 0 1 2 0 0 0 0 1 1 1 1 ...
node 3 view 20 40
link 1 0 2 0
link 2 0 3 0

[graph]
# node <id> <kind> <x> <y> <parameters in node_spec order>
node 1 time 20 60 1 0
node 5 output 800 120
node 6 neighborhood 20 480
node 8 growth 250 460 0 0 0.15 0.015 0.1
# link <from node> <output port> <to node> <input port>
link 1 0 2 0
link 6 0 8 0
```

Node kinds are `time`, `oscillator`, `constant`, `kernel`, `environment`, `output` (the
Simulation node) and, since version 2, the cell rule nodes `density`, `neighborhood`, `growth`,
`add`, `multiply`, `clamp`, `compare`, `select`, `life`, since version 4 `nutrient` and, since
version 5, the reusable kernels `conv_kernel` (Kernel K) and `convolve`.
Parameters follow `node_spec()` in
`src/node_graph.cpp`: Time `speed start`; Oscillator `wave period phase amplitude offset` (wave
0–3 = sine, triangle, square, saw); Constant `value`; Kernel `radius_delta peak_shift sharpness`;
Environment `pattern strength scale angle offset` (pattern 0–4 = uniform, spot, gradient,
stripes, ripples; angle in degrees); Output none; Density and Neighborhood none; Growth `source
function mu sigma dt` (source 0 = Simulation panel, 1 = this node; function 0 = gaussian,
1 = polynomial); Nutrient none; Add and Multiply `a b`; Clamp `x low high`; Compare `test a b` (test 0–5 =
`>`, `≥`, `<`, `≤`, `=`, `≠`); Select `condition if_true if_false`; Life `birth_mask
survive_mask` (whole numbers 0–511); Kernel K `radius core rings ring1 ring2 ring3 normalization
shift_x shift_y` (radius 1–32; core 0–3 as in `[kernel]`; rings 1–3, the weights past it are
stored but unused; normalization 0–3 = none, sum, positive, abs_sum; shifts in cells, −16 to 16,
with radius + the larger shift at most 32); Convolve none. Ports are numbered from 0 in the
order listed in `node_graph.md`: Environment outputs 0 Environment, 1 Field; Simulation inputs
0 Kernel, 1 Environment, 2 Next state, 3 Next nutrient (version 4); Growth inputs 0 U, 1 μ, 2 σ
and outputs 0 G, 1 dt; Density and Nutrient outputs 0 value, 1 Field (version 5); Kernel K
output 0 K; Convolve inputs 0 Field, 1 Kernel and output 0 U.

`[display_graph]` (version 3) has the same `node` and `link` lines with its own ids. Its kinds
are `fields`, `ramp`, `color`, `mix`, `view` and the Value kinds `time`, `oscillator`,
`constant`. Parameters: Fields none; Ramp `low high stops` then 10 stops of `position red green
blue` (all 10 are written; only the first `stops`, 2–10, are used); Color `red green blue`; Mix
`mode factor` (mode 0–3 = mix, add, multiply, screen); View none. Ports: Fields outputs
0 Density, 1 Neighborhood, 2 Growth, 3 Nutrient (version 4); Ramp input 0 Value; Mix inputs 0 A, 1 B, 2 Factor; View
input 0 Color.

## Versioning

`kProjectVersion` is the version written. A build reads every version up to its own and
rejects newer ones with a message rather than guessing. A change that existing readers would
misread (new required keys, changed meanings, new node parameters) gets a new version, with the
parser keeping older versions readable; `tests/test_project.cpp` pins hand-written version 1
files.

**Version 2** (cell rules) added node kinds and ports only; every key and section is as in
version 1, and a version 1 file reads unchanged. Its graph has no Next state link, so it runs
the **built-in Lenia update**, exactly as it did when saved (the backward-compatible default;
nothing is rewritten on open). To edit that update as nodes, select the Simulation node and
press *Wire the built-in Lenia rule as nodes* (`lenia::wire_lenia_rule`): the result is
identical, an Environment linked into Simulation moves into the rule, and saving writes the
current version. A build from before rule nodes rejects version 2 files with the "newer than
this workbench reads" message.

**Version 3** (display graphs) added the `[display_graph]` section and its node kinds; every
other section, key and node is as in version 2. A version 1 or 2 file has no display graph and
opens with a lone View node, so the view shows its `[display]` settings exactly as when it was
saved; saving it writes version 3 with that one-node display graph. `tests/test_display_graph.cpp`
covers both directions; builds before version 3 reject version 3 files as newer.

**Version 4** (two state fields, `state_fields.md`) added the required `[nutrient]` section, the
`nutrient` node kind, the Simulation's Next nutrient input, Fields' Nutrient output and the
`nutrient` display key. A version 1–3 file opens with a uniform nutrient of 1, which its graphs
never read, and the default nutrient display range; it runs exactly as before. Saving it writes
version 4. `tests/test_state_fields.cpp` covers the roundtrip and the version gates.

**Version 5** (reusable kernels, `node_graph.md` *Several kernels*) added the `conv_kernel` and
`convolve` node kinds and the Field outputs of Density A and Nutrient N that feed them; every
section and key is as in version 4. A version 1–4 file has none of them and runs exactly as
before; saving it writes version 5. An older file naming either kind is rejected with the
version it needs. `tests/test_convolve_node.cpp` covers the roundtrip and the gate.
