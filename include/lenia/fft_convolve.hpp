#pragma once

#include "lenia/convolve.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lenia {

// How the GPU computes the neighborhood U = K * A (convolve's toroidal sum).
enum class ConvolutionMethod : std::uint32_t {
    // One invocation per cell sums the kernel's nonzero taps: cost grows with the kernel's
    // nonzero cells. Any grid size. The default.
    Direct = 0,
    // Radix-2 FFT convolution (shaders/fft_convolve.comp): cost grows with the grid only, so it
    // pays off for dense, large kernels. Power-of-two grids only (fft_size_supported); others
    // fall back to Direct. Matches Direct up to float rounding, not bit for bit.
    Fft = 1,
};

// Shortest and longest FFT line (grid width or height) shaders/fft_convolve.comp handles. One
// workgroup transforms one line in shared memory, 8 bytes per element, so a device's limit is
// usually lower: see fft_max_size.
constexpr std::size_t kFftMinSize = 8;
constexpr std::size_t kFftMaxSize = 8192;

// Longest supported line on a device with `shared_bytes` of compute shared memory
// (maxComputeSharedMemorySize): the largest power of two with 8 bytes per element that fits,
// at most kFftMaxSize; 0 if not even kFftMinSize fits.
std::size_t fft_max_size(std::size_t shared_bytes);

// Whether a width x height grid can use the FFT path: both powers of two in
// [kFftMinSize, max_size].
bool fft_size_supported(std::size_t width, std::size_t height, std::size_t max_size = kFftMaxSize);

// log2 of a power of two.
std::uint32_t fft_log2(std::size_t n);

// exp(-2 pi i t / n) for t in [0, n / 2), computed in double, as interleaved float (re, im)
// pairs: the twiddle table shaders/fft_convolve.comp reads. Every line of length m <= n (a
// power of two dividing n) uses entries t * (n / m). n must be a power of two >= 2.
std::vector<float> fft_twiddles(std::size_t n);

// Complex values in a half spectrum: (width / 2 + 1) * height. The input is real, so the other
// columns are conjugates of these.
std::size_t fft_spectrum_size(std::size_t width, std::size_t height);

// convolve(grid, kernel) computed as shaders/fft_convolve.comp does, in float, step for step:
// the CPU reference of the GPU's FFT path (tests/test_fft_convolve.cpp checks it against
// convolve). Equal to convolve up to float rounding; kernels larger than the grid wrap the same
// way. Throws std::invalid_argument if the kernel has an even or zero dimension, a cell count
// does not match width * height, or !fft_size_supported(grid.width, grid.height).
Grid convolve_fft(const Grid& grid, const Grid& kernel);

// The passes of shaders/fft_convolve.comp (FftPushConstants::pass). One Lenia generation is
// RowsForward, ConvolveColumns, RowsInverse; KernelRowsForward then KernelColumns compute the
// kernel's spectrum, needed again only when the kernel or the grid size changes.
enum class FftPass : std::uint32_t {
    RowsForward = 0,       // src rows (two per workgroup) -> spec, half spectrum of each row
    KernelRowsForward = 1, // the kernel folded onto the grid's torus -> spec
    KernelColumns = 2,     // spec columns -> khat, scaled by 1 / (width * height)
    ConvolveColumns = 3,   // spec columns, times khat, back: spec
    RowsInverse = 4,       // spec -> U; dst = U, or with fuse the next Lenia state
};

// Must match the push_constant block in shaders/fft_convolve.comp.
struct FftPushConstants {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t log2_width;
    std::uint32_t log2_height;
    std::uint32_t log2_twiddles; // log2 of the n the twiddle table was made for
    std::uint32_t kwidth;
    std::uint32_t kheight;
    std::uint32_t pass;  // FftPass
    std::uint32_t fuse;  // RowsInverse: 1 writes the next state, 0 writes U
    float mu;
    float sigma;
    float dt;
    std::uint32_t growth; // lenia::GrowthFunction
};

// Workgroups to dispatch (x; y and z are 1) for a pass on a width x height grid.
std::uint32_t fft_pass_groups(FftPass pass, std::size_t width, std::size_t height);

} // namespace lenia
