# GPU benchmark (`lenia_bench`)

`lenia_bench` is a headless, reproducible baseline for the Vulkan compute step: batched Lenia
generations through `VulkanStepper`, with direct convolution (`shaders/lenia_step.comp`) and/or
FFT convolution (`shaders/fft_convolve.comp`, see below). It opens no window and does not touch
the workbench.

## Build and run

Built with `LENIA_ENABLE_VULKAN=ON` (the `build-vk` directory). Measure in Release:

```
cmake --build build-vk --config Release --target lenia_bench
build-vk\Release\lenia_bench.exe
build-vk\Release\lenia_bench.exe --csv > bench.csv
```

| Option           | Default          | Meaning |
|------------------|------------------|---------|
| `--sizes LIST`   | `256,512,1024`   | grid sizes, `N` or `WxH`, comma-separated |
| `--kernels LIST` | `small,orbium,large` | which kernels (below) |
| `--methods LIST` | `direct,fft`     | convolution methods; each case runs once per method |
| `--steps N`      | 32               | generations per timed batch (one GPU submission) |
| `--warmup N`     | 3                | untimed batches per case before sampling |
| `--samples N`    | 11               | timed batches per case; the median is reported |
| `--csv`          | off              | CSV on stdout (header lines start with `#`) |

Exit codes: 0 measured and verified; 1 a verification or Vulkan error; 2 bad usage; 77 skipped
(no Vulkan device, or no timestamp queries on its compute queue).

## What is measured

Every kernel uses the bundled Orbium's core (polynomial), single ring and polynomial growth
(μ 0.15, σ 0.015, dt 0.1), sum-normalized; only the radius changes:

| Kernel   | Radius | Size  |
|----------|--------|-------|
| `small`  | 4      | 9×9   |
| `orbium` | 13     | 27×27 (the bundled Orbium as is) |
| `large`  | 32     | 65×65 |

The direct shader skips zero weights, so the `taps` column (nonzero weights) is its work per
cell. The FFT's work depends on the grid only.

For each kernel × grid size: `--warmup` batches, then `--samples` batches of `--steps`
generations, all from the same deterministic soup (values in [0, 0.3)). Each batch is one
submission: upload, `steps` dispatches, readback. **GPU time** comes from two timestamp queries
written at the bottom of the pipe after the upload and after the last dispatch, so it covers the
compute only, not upload, submission or readback. Grid and kernel live in device-local memory,
as in the workbench. The table reports per case:

- `ms/gen`: median batch GPU time ÷ steps; `ms/batch med`, `min`, `max` over the samples;
- `Mcell/s`: million cell updates per second at the median;
- `wall ms`: median host time per batch, *including* transfers and submission (context only);
- `xform ms` (CSV `kernel_transform_ms`): FFT only, the one-time GPU cost of the kernel's
  spectrum for this kernel and grid size, measured on its first use; not in the timed batches;
- `verify`, see below.

For a fair comparison both methods run the same cases from the same soup with the same
`--steps`, `--warmup` and `--samples`, and their GPU time covers the same span: the
generations of a batch, after the upload. The FFT kernel spectrum is cached across batches (the
kernel does not change), so it is reported apart instead of being amortized into `ms/gen`;
in the workbench it is recomputed only when the kernel or the world size changes. The FFT rows
of grids it does not support (not power-of-two) say `n/a` and are not run.

The CSV has a `method` column (after `kernel`) and `kernel_transform_ms` (before `verify`);
otherwise its columns are as before.

The header records device, type, vendor/device IDs, driver and Vulkan versions, timestamp
period, growth, method and build type; keep it with the numbers. Without timestamp support the
cases are still verified, timing columns say `skipped`, and the exit code is 77; there is no
wall-time fallback.

## Verification

Before timing, each case checks the GPU against the existing CPU reference, as
`tests/test_vulkan_parity.cpp` does: one generation against `lenia::step_lenia` with tolerance
1e-4 (at the case's size, or 256×256 when the CPU would be slow; the shader is the same), and a
batch of two generations against two single GPU steps, bit for bit. Every timed sample must also
give bit-identical cells. FFT rows are additionally checked against direct on the GPU at the
case's full size (one generation, tolerance 1e-4), and must actually have run the FFT. Any
failure marks the row and exits 1. Run the full parity suite too:

```
ctest --test-dir build-vk -C Release -R "vulkan_parity|fft_convolve" --output-on-failure
```

## FFT convolution

`VulkanStepper::set_convolution(lenia::ConvolutionMethod::Fft)` (and the workbench's
Playback → Convolution choice) computes U = K ∗ A as a product of spectra, in a radix-2 FFT of
our own (`shaders/fft_convolve.comp`; VkFFT was not used because it compiles its shaders at
run time with glslang, an extra dependency for every build). The default stays **Direct**.

- **Semantics**: the same toroidal sum as `lenia::convolve` and the direct shader: kernel tap
  (kx, ky) lands at offset (kx − cx, ky − cy), so asymmetric kernels keep their orientation,
  and kernels larger than the grid wrap (taps on the same cell add up). The kernel is used as
  given, so its normalization is unchanged. Results equal Direct up to float rounding (the
  parity tests use 1e-4 after a step), not bit for bit; FFT batches are deterministic.
- **Supported grids**: both sides powers of two from 8 to the device's limit
  (`VulkanDeviceInfo::fft_max_size`, from its shared memory: 4096 with 32–48 KiB, 2048 with
  16 KiB). Anything else, e.g. a 37×23 world or an old snapshot of odd size, silently runs Direct;
  `VulkanStepper::last_convolution()` tells which ran. `step_rule` always runs Direct.
- **Cost**: per generation three dispatches (row FFTs of row pairs, column FFT × kernel
  spectrum × inverse, inverse row FFTs fused with the growth update), independent of the
  kernel. The kernel's spectrum takes two more dispatches, only when the kernel or grid size
  changes.
- **Workbench**: FFT also requires the kernel to have at least `LeniaView::kFftMinTaps`
  (100) nonzero weights, an estimate of where it starts to pay off; tune it from this
  benchmark's `small` rows. The status bar says when generations switch between FFT and Direct.

`lenia::convolve_fft` is the same algorithm on the CPU, step for step, tested against
`lenia::convolve` in `tests/test_fft_convolve.cpp` (in the CPU-only build too).

## Reproducibility

- Close other GPU work; plug in laptops; prefer a fixed power plan. Compare runs on the same
  driver only.
- Variation between `min` and `max` shows noise; rerun with more `--samples` if it is large.
- Windows resets the GPU if one submission runs longer than ~2 s (TDR). The tool warns above
  1 s per batch; lower `--steps` for big grids with the large kernel (e.g. `--sizes 2048 --steps 8`).
