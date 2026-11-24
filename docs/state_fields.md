# Two state fields: Density A and Nutrient N

The simulation has two named scalar fields per cell (`include/lenia/world.hpp`):

| Field       | Read by                              | Written by |
|-------------|--------------------------------------|------------|
| Density A   | the kernel (→ Neighborhood U), the Density node, Fields.Density, Convolve nodes | the Simulation node's **Next state** (or the built-in Lenia update) |
| Nutrient N  | the **Nutrient N** node, Fields.Nutrient, Convolve nodes | the Simulation node's **Next nutrient** |

The active kernel convolves only the density. To give either field a neighborhood of its own,
link its node's **Field** output into a **Convolve** node with a **Kernel K** (`node_graph.md`,
*Several kernels*): the density and the nutrient can each have independent kernels, e.g. a
nutrient that diffuses (`next N = K ∗ N` with a small Sum-normalized kernel) while the density
follows Lenia.

## Simultaneous update

A generation computes every cell's next A and next N from the **current** A, N and U, so neither
update sees the other's result. A rule `A' = N, N' = A` exchanges the two fields exactly
(`tests/test_state_fields.cpp`). A node that feeds both inputs is evaluated once per cell.

- **Next nutrient unlinked:** N is kept bit for bit. Every experiment from before the nutrient
  runs exactly as it did.
- **No rule (built-in Lenia update):** N is kept.
- **Next nutrient linked without Next state:** this is an error ("Next nutrient is linked but
  Next state is not"), and the simulation does not step. Use the *Wire the built-in Lenia rule
  as nodes* button first.

## Starting field

Each experiment has a `[nutrient]` section (`projects.md`) and the World panel's *Nutrient
start* and *Nutrient level* controls:

- `uniform`: `level` everywhere (the default; 1).
- `random`: independent values in [0, level), from the experiment's random seed (mt19937 seeded
  with `random_seed XOR 0x4e555452`, top 24 bits of each draw). The values are identical on every
  platform.
- `gradient`: 0 at the left column, rising linearly to `level` at the right one.

*Reset (R)* rebuilds both fields: the density from its seed and the nutrient from this setting.

## The nutrient trail demo

The node editor's *Density + Nutrient trail (two state fields)* example, the
*Nutrient trail (Density + Nutrient)* button or `GraphPreset::NutrientTrail` loads Orbium on a uniform nutrient of 1
with this rule:

```
A' = clamp(A + dt · (G(U) + 0.03 · (N − 1)), 0, 1)       starvation where nutrient is low
N' = clamp(N + 0.002 · (1 − N) − 0.02 · A · N, 0, 1)     recovery − uptake
```

The creature eats the nutrient where it passes and leaves a dark trail that slowly grows back.
At N = 1 the density update is the wired Lenia rule bit for bit, so the first generation equals
plain Orbium. The display uses the `DensityNutrient` preset: a soil ramp (brown → green) on the
nutrient, screened with a creature ramp on the density.

## Inspecting and painting

- **Field combo → Nutrient N** shows the nutrient with its own palette and range (Inferno 0–1 by
  default).
- **Inspect fields** (or **Alt+click** a cell) reads both fields back once and reports min, max,
  mean and mass of A and N, plus both values at the clicked cell.
- **Paint field** chooses which field the world brush paints.
- **Display graph:** Fields has a *Nutrient* output (port 3).

## GPU layout

Every cell buffer holds two planes, `[density | nutrient]`, with the nutrient at offset
`width × height`. The rule program header word 3 is the nutrient register (or `kNoRegister`).
`VulkanStepper::step_rule(WorldState, …)` runs both fields; `tests/test_vulkan_parity.cpp` and
`tests/test_workbench_smoke.cpp` compare them with the CPU.

## Files

Experiment format version 4 adds `[nutrient]`, the `nutrient` node kind, the Simulation's Next
nutrient input (port 3), Fields' Nutrient output (port 3) and the `nutrient` display key.
Snapshot version 2 stores the nutrient after the density. Older files still open: their
nutrient is the uniform field of 1 that their graphs never read.

## Limits

- The two field names are fixed. They cannot be renamed and no third field can be added.
- The nutrient does not diffuse and has no kernel of its own.
