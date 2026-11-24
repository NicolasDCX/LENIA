# LENIA

A C++20 implementation of [Lenia](https://github.com/Chakazul/Lenia), the continuous cellular
automaton, with a CPU reference core, an optional Vulkan compute backend and an optional Qt 6
workbench for editing kernels, rules and displays while the simulation runs.

## Contents

| Path              | What it is |
|-------------------|------------|
| `include/`, `src/` | `lenia_core`: convolution (direct and FFT), kernels, growth, rules, node graphs, pattern, experiment and snapshot files. No dependencies beyond the C++ standard library |
| `shaders/`        | GLSL compute and display shaders, compiled to SPIR-V at build time |
| `apps/workbench/` | `lenia_workbench`, the Qt 6 Widgets + Vulkan workbench |
| `apps/bench/`     | `lenia_bench`, a headless GPU benchmark |
| `tests/`          | CTest suite: CPU tests, CPU/GPU parity, workbench smoke and undo tests |
| `patterns/`       | Bundled creature data (Orbium) and its license |
| `docs/`           | Node graph, display graph, experiment and snapshot formats, benchmark |

## Building

Requires CMake 3.20+ and a C++20 compiler.

CPU core and tests only:

```
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Vulkan backend, parity test and benchmark (needs the Vulkan SDK, including `glslc` or
`glslangValidator`):

```
cmake -S . -B build-vk -DLENIA_ENABLE_VULKAN=ON
cmake --build build-vk --config Release
```

Workbench (also needs Qt 6.2+ with Vulkan support):

```
cmake -S . -B build-editor -DLENIA_ENABLE_VULKAN=ON -DLENIA_BUILD_WORKBENCH=ON ^
      -DCMAKE_PREFIX_PATH=C:/Qt/6.8.3/msvc2022_64
cmake --build build-editor --config Release
cmake --install build-editor --config Release --prefix dist
```

On Windows, `windeployqt` copies the Qt runtime next to the executable after each build
(`-DLENIA_WORKBENCH_DEPLOY=OFF` to disable). GPU tests report as skipped when no Vulkan device
is available.

## Documentation

- [Node graph](docs/node_graph.md): per-cell rules, kernel and environment modulation, and
  reusable Kernel K / Convolve nodes for several neighborhoods on either field
- [Display graph](docs/display_graph.md): how the viewport colors cells
- [Two state fields](docs/state_fields.md): Density + Nutrient, and the nutrient trail demo
- [Undo and redo](docs/undo.md): one history for graph, kernel and growth edits
- [Experiment files](docs/projects.md) (`.leniaexp`) and [snapshots](docs/snapshots.md) (`.leniasnap`)
- [GPU benchmark](docs/benchmark.md) and FFT convolution
- [Pattern files](patterns/README.md)

## License

LENIA is licensed under the GNU General Public License, version 3 only (SPDX:
`GPL-3.0-only`); see [`LICENSE`](LICENSE).

The bundled Orbium pattern (`patterns/orbium.lenia`) is from
[Chakazul/Lenia](https://github.com/Chakazul/Lenia) by Bert Wang-Chak Chan and is distributed
under the MIT License, Copyright (c) 2018 Bert Chan; see
[`patterns/ORBIUM-LICENSE.txt`](patterns/ORBIUM-LICENSE.txt). Lenia is described in
B. W.-C. Chan, "Lenia — Biology of Artificial Life", *Complex Systems* 28(3), 2019.
