# Lenia pattern files

The workbench (`lenia_workbench`, "Load pattern file…") and `lenia::load_pattern_file` read a
small text format. Keep each file's source and license next to its cells, and add any bundled
file and its license to `LENIA_PATTERN_FILES` in `CMakeLists.txt` so they ship with the
workbench.

```
# comment
name    Orbium unicaudatus
source  <URL or citation of the original data>
license <license of the original data>
R       13            # kernel radius in cells (integer, at most 64 in the workbench)
T       10            # or: dt 0.1
mu      0.15
sigma   0.015
rings   1             # signed ring weights, innermost first (Lenia's beta)
core    polynomial    # exponential | polynomial | step | gaussian
growth  polynomial    # gaussian | polynomial (default gaussian)
cells                 # must be the last key
[[0, 0, 0.1, ...],
 [0, 0.08, 0.24, ...],
 ...]
```

The cell array is rows of numbers, top row first. It may be pasted as a Python / JSON nested
list (each `]` ends a row) or written as plain whitespace-separated rows (each line is a row).
Short rows are padded with zeros.

Growth functions, with z = (U − μ) / σ:

| `growth`     | G(U)                          | upstream Lenia |
|--------------|-------------------------------|----------------|
| `gaussian`   | 2·exp(−z²/2) − 1              | `gn = 2`       |
| `polynomial` | 2·max(0, 1 − z²/9)⁴ − 1       | `gn = 1`       |

Kernel cores map to upstream `kn` the same way: `polynomial` is `kn = 1`, `exponential` is
`kn = 2`, `step` is `kn = 3`. When converting an upstream creature, copy both `kn` and `gn`:
a creature found with polynomial growth generally does not survive Gaussian growth.

## Bundled: Orbium (`orbium.lenia`)

Orbium unicaudatus, entry O2u of `Python/animals.json` in
<https://github.com/Chakazul/Lenia>, by Bert Wang-Chak Chan, decoded from its RLE cell string.
Its rule is R = 13, T = 10, μ = 0.15, σ = 0.015, β = [1], polynomial core and polynomial
growth. Described in B. W.-C. Chan, "Lenia — Biology of Artificial Life", *Complex Systems*
28(3), 2019.

The data is under the MIT license, Copyright (c) 2018 Bert Chan; the full text is in
`ORBIUM-LICENSE.txt`, which must be distributed with the file. The workbench compiles both
files in as Qt resources, copies them to `patterns/` next to the executable after each build,
and `cmake --install` puts them in `bin/patterns/`.

The workbench starts with this Orbium running; "Restore Orbium" (or O) reloads it.
`tests/test_orbium.cpp` checks that it loads with the upstream rule and that it survives and
glides on the CPU, and `tests/test_vulkan_parity.cpp` steps it on the GPU against the CPU.
