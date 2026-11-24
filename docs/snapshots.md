# Simulation snapshots (`.leniasnap`)

*File → Save snapshot… (Ctrl+Alt+S)* stores the **running world** of `lenia_workbench` — its
density, its generation and the experiment that drives it — so *File → Open snapshot…
(Ctrl+Alt+O)* can continue it exactly. Experiment files (`projects.md`) are unchanged and
separate: they store the setup only and restart at generation 0.

|                          | Experiment `.leniaexp`         | Snapshot `.leniasnap` |
|--------------------------|--------------------------------|-----------------------|
| Kernel, growth, graphs, display, seed | yes               | yes (an experiment, embedded unchanged) |
| World density            | no (rebuilt from the seed)     | yes, every cell bit for bit |
| World nutrient (`state_fields.md`) | no (rebuilt from `[nutrient]`) | yes, every cell bit for bit (version 2) |
| Generation               | no (opens at 0)                | yes |
| Hand-painted cells       | no                             | yes (they are part of the density) |

The format is implemented in `lenia::Snapshot` (`include/lenia/snapshot.hpp`,
`src/snapshot.cpp`), without Qt; `tests/test_snapshot.cpp` covers it.

## Exact continuation

A workbench generation depends on the density, the experiment and the generation number: the
node graph (kernel warp, environment, rule constants) and the display graph are pure functions
of the generation (`node_graph.md`). A snapshot stores all three, so opening it and stepping
computes **the same next generations, bit for bit, on the same GPU and driver**, time
modulation included. Opening a snapshot:

- pauses the simulation and shows the stored generation (Space continues it);
- uses the stored experiment for everything, including *Reset (R)*, which restarts that
  experiment from its seed at generation 0;
- recomputes the Neighborhood and Growth fields for the loaded state. While running, the view
  shows those of the state each step started from, so right after loading they can differ from
  what was on screen when saving; the density, and every later generation, do not.

Not stored: UI-only settings (brushes, Step ×N, generations per frame, zoom, docks). After
opening a snapshot, *Save experiment* asks for a file name.

## Capturing from the GPU

The world lives only in GPU buffers, and playback never reads it back. Saving a snapshot asks
`LeniaView::capture_world` for one readback: the next frame, after its uploads (painted cells,
seeds) and steps, records a pipeline barrier (transfer + compute writes → transfer read), a copy
of the density into a host-visible buffer and a barrier to host reads. Once that frame is
submitted, the CPU waits for the graphics queue, maps the buffer and copies it out — a one-off
stall only when a snapshot is saved. The callback runs inside that frame, so the experiment
recorded with the density is exactly the setup its next generation will use; writing the file
happens after the frame. Saving fails with a message, changing nothing, if the viewport is
minimized, the device fails or is reset first, or another snapshot is still being captured.

## Opening is all or nothing; saving is atomic

The whole file is read, parsed and checked before anything changes: format, byte counts,
checksum, the embedded experiment (every check of `projects.md`, including the workbench's
limits), and a density of the experiment's world size whose cells are all finite. On any error
a message says what is wrong and the current experiment and world are untouched.

Saving writes through `QSaveFile` (a temporary file renamed over the target once complete), so
a failed save leaves an existing file intact, and the workbench parses its own output again
before writing and refuses to save anything it could not reopen.

## Format, version 2

A short text header, then binary data; lines end with `\n` only, and the file must not be
edited (the checksum fails).

```
lenia-snapshot 2
# Lenia workbench simulation snapshot (docs/snapshots.md); binary data follows, do not edit
generation 1234
experiment <N>
<N bytes: the experiment exactly as a .leniaexp file holds it (projects.md)>
density <W> <H> f32le
<W × H × 4 bytes: rows top first, each cell an IEEE 754 float32, little-endian>
nutrient <W> <H> f32le
<W × H × 4 bytes, the same layout>

checksum fnv1a64 <16 lowercase hex digits>
```

Version 1 has no nutrient block. Opening one gives the nutrient its experiment's starting field
(a uniform 1 for every experiment before version 4), which nothing in its graph reads, so it
continues exactly as before.

- `generation`: 0 to 2⁶⁴ − 1, the generation the density is at (the next step computes
  `generation + 1`).
- `W` and `H` of both fields must equal the experiment's world size; cells may be any finite
  value (rules can leave [0, 1]); NaN and infinities are rejected.
- Each field is followed by a `\n`; `checksum` is 64-bit FNV-1a over every byte before the
  checksum line. Nothing may follow it. Lines starting with `#` are comments in the header.

`kSnapshotVersion` is the version written; a build reads every version up to its own and
rejects newer ones. The embedded experiment carries its own `lenia-project` version, so
experiment format changes need no new snapshot version.

## Tests

- `tests/test_snapshot.cpp` (CPU): bit-for-bit roundtrip (−0, denormals, values above 1,
  generations beyond 32 bits); a snapshot of a time-modulated run (breathing kernel, pulsing
  environment) continued from its generation equals the uninterrupted run, while restarting
  the counter at 0 diverges; damaged, truncated, inconsistent and malformed files are rejected.
- `tests/test_workbench_smoke.cpp` (built with `LENIA_BUILD_WORKBENCH`, label `gui`): opens the
  real workbench window on a Vulkan device, opens a known experiment paused, checks the GPU seed
  against the CPU, steps it, changes the display graph while paused and checks that the
  rendered viewport (`QVulkanWindow::grab`) changes while generation and density stay
  bit-identical, saves a snapshot, refuses a damaged one without touching the world, reopens
  the snapshot while running (paused at its generation) and steps it to the original run's
  result bit for bit, and checks that playback performs no readbacks. It then runs the nutrient
  trail demo, compares a GPU generation of both fields with the CPU, and reopens a two-field
  snapshot bit for bit. `tests/test_state_fields.cpp` covers two-field roundtrips and version 1
  files. It needs a desktop
  session; without a Vulkan device it reports skipped (77). Run it with
  `ctest -C Release -L gui --output-on-failure`.
