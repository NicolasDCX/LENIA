#pragma once

#include "lenia/fft_convolve.hpp"
#include "lenia/rule.hpp"
#include "lenia/step.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace lenia {

// The device a VulkanStepper runs on, for labeling benchmark results (docs/benchmark.md).
struct VulkanDeviceInfo {
    std::string name;
    std::string type;                 // "discrete", "integrated", "virtual", "cpu" or "other"
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint32_t driver_version = 0; // raw VkPhysicalDeviceProperties::driverVersion (vendor encoding)
    std::uint32_t api_version = 0;    // VK_MAKE_API_VERSION encoding
    bool timestamps = false;          // the compute queue supports timestamp queries
    double timestamp_period_ns = 0.0; // nanoseconds per timestamp tick
    bool device_local = false;        // grid and kernel buffers live in device-local memory
    std::size_t fft_max_size = 0;     // longest grid side the FFT path handles (0: none)
};

// Runs step_lenia on a Vulkan compute device (shaders/lenia_step.comp).
//
// Results match the CPU step_lenia, for both growth functions, up to float rounding: the GPU
// may fuse multiply-adds and uses its own exp, so compare with a tolerance rather than exactly.
//
// The grid, kernel and rule program are copied through a host-visible staging buffer into
// device-local buffers (when the device has such memory), where every generation of a batch
// reads and writes them.
//
// The constructor picks a discrete GPU with a compute queue if there is one, otherwise any
// compute-capable device, and throws std::runtime_error if Vulkan or such a device is
// unavailable. Not thread-safe; use one stepper per thread.
class VulkanStepper {
public:
    VulkanStepper();
    ~VulkanStepper();
    VulkanStepper(VulkanStepper&&) noexcept;
    VulkanStepper& operator=(VulkanStepper&&) noexcept;
    VulkanStepper(const VulkanStepper&) = delete;
    VulkanStepper& operator=(const VulkanStepper&) = delete;

    // Same contract as lenia::step_lenia. Throws std::invalid_argument if the kernel has an
    // even or zero dimension or a grid's cell count is not width * height, and
    // std::runtime_error if a Vulkan call fails or the grid exceeds device limits.
    Grid step_lenia(const Grid& grid, const Grid& kernel, const GrowthParams& params);

    // Advances `grid` by `generations` Lenia generations in one GPU submission: the grid is
    // uploaded once, ping-pongs between two device buffers, and is read back once. Equivalent
    // to calling step_lenia `generations` times, up to float rounding (which can compound
    // across generations). Zero generations returns a copy of `grid`. Throws as step_lenia.
    Grid step_lenia(const Grid& grid, const Grid& kernel, const GrowthParams& params, std::size_t generations);

    // lenia::step_rule on the device (shaders/rule_step.comp, the workbench's interpreter
    // lenia_rule.glsl), `generations` times with the same program, in one submission. Matches
    // the CPU up to float rounding; rules made of exact operations (the Game of Life on the
    // Moore kernel) match exactly. Throws as step_lenia.
    Grid step_rule(const Grid& grid, const Grid& kernel, const RuleProgram& program, std::size_t generations = 1);
    // The same for both state fields (lenia::step_rule on a WorldState): the density and the
    // nutrient travel together in one buffer per generation and update simultaneously. Throws
    // std::invalid_argument also if the fields differ in size.
    WorldState step_rule(const WorldState& world, const Grid& kernel, const RuleProgram& program,
                         std::size_t generations = 1);

    // How step_lenia computes the neighborhood; Direct (shaders/lenia_step.comp) by default.
    // With Fft, power-of-two grids the device supports (fft_supported) go through
    // shaders/fft_convolve.comp: same results up to float rounding (so compare with a
    // tolerance, as against the CPU), and batches stay deterministic. Other grids silently use
    // Direct; last_convolution tells which ran. step_rule always uses Direct.
    //
    // The kernel's spectrum is computed on the device and kept: later calls with an equal
    // kernel (same size and weights) on a grid of the same size reuse it.
    void set_convolution(ConvolutionMethod method);
    ConvolutionMethod convolution() const;
    bool fft_supported(std::size_t width, std::size_t height) const;
    // The method the last step_lenia call used; Direct after step_rule.
    ConvolutionMethod last_convolution() const;

    const std::string& device_name() const;
    const VulkanDeviceInfo& device_info() const;

    // Device time, in milliseconds, of the last step_lenia / step_rule call's generations: from
    // the end of its upload (and of the FFT kernel transform, if one ran) to the end of its last
    // generation, measured with GPU timestamp queries, so it excludes the host copies,
    // submission and readback. Empty if the device has no timestamps (device_info().timestamps),
    // the call dispatched nothing (zero generations or an empty grid), or it threw.
    std::optional<double> last_gpu_ms() const;
    // Device time of the FFT kernel transform the last call ran before its generations (a new
    // kernel or grid size); empty if it reused the cached spectrum, used Direct, or as above.
    std::optional<double> last_kernel_transform_ms() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lenia
