// lenia_bench: headless, reproducible benchmark of batched Lenia generations on the Vulkan
// compute backend (VulkanStepper: direct convolution in shaders/lenia_step.comp, or FFT
// convolution in shaders/fft_convolve.comp). Times each batch on the GPU with timestamp
// queries, after warm-up batches, and reports the median of repeated samples. Each case is
// first checked against the CPU step_lenia (and FFT against direct on the GPU). See
// docs/benchmark.md.

#include "lenia/kernel_shape.hpp"
#include "lenia/pattern.hpp"
#include "lenia/step.hpp"
#include "lenia/vulkan_step.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Return code for "nothing measured": no Vulkan device, or no GPU timestamps.
constexpr int kSkip = 77;

// GPU may fuse multiply-adds and uses its own exp; as tests/test_vulkan_parity.cpp.
constexpr float kTol = 0.0001f;

// Largest CPU check, in kernel taps times cells; bigger cases are checked on a 256x256 grid.
constexpr double kVerifyBudget = 3e8;
constexpr std::size_t kVerifyFallback = 256;

// A batch longer than this risks the Windows GPU timeout (TDR, 2 s by default).
constexpr double kTdrWarnMs = 1000.0;

struct Size {
    std::size_t w = 0, h = 0;
};

// Kernels share Orbium's core, rings and growth, and differ in radius. Radius 0 means the
// bundled Orbium's own (13).
struct KernelCase {
    std::string name;
    int radius = 0;
};

const KernelCase kKernels[] = {{"small", 4}, {"orbium", 0}, {"large", 32}};

const char* method_name(lenia::ConvolutionMethod m) { return m == lenia::ConvolutionMethod::Fft ? "fft" : "direct"; }

struct Options {
    std::vector<Size> sizes{{256, 256}, {512, 512}, {1024, 1024}};
    std::vector<KernelCase> kernels{std::begin(kKernels), std::end(kKernels)};
    std::vector<lenia::ConvolutionMethod> methods{lenia::ConvolutionMethod::Direct, lenia::ConvolutionMethod::Fft};
    std::size_t steps = 32;
    std::size_t warmup = 3;
    std::size_t samples = 11;
    bool csv = false;
};

void usage(std::FILE* out) {
    std::fprintf(out,
                 "Usage: lenia_bench [options]\n"
                 "  --sizes LIST    grid sizes, N or WxH, comma-separated (default 256,512,1024)\n"
                 "  --kernels LIST  small (R4), orbium (R13), large (R32) (default all)\n"
                 "  --methods LIST  direct, fft (default both; fft needs power-of-two sides)\n"
                 "  --steps N       generations per timed batch, one submission (default 32)\n"
                 "  --warmup N      untimed batches before sampling (default 3)\n"
                 "  --samples N     timed batches; the median is reported (default 11)\n"
                 "  --csv           CSV on stdout instead of a table\n"
                 "Exit codes: 0 measured and verified, 1 verification or Vulkan error, 2 bad usage,\n"
                 "77 skipped (no Vulkan device, or no GPU timestamps on its compute queue).\n");
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (true) {
        const std::size_t end = s.find(sep, start);
        parts.push_back(s.substr(start, end - start));
        if (end == std::string::npos) return parts;
        start = end + 1;
    }
}

std::size_t parse_count(const std::string& s, std::size_t min) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("not a number: '" + s + "'");
    }
    const unsigned long long n = std::stoull(s);
    if (n < min || n > 1000000) throw std::invalid_argument("out of range: " + s);
    return static_cast<std::size_t>(n);
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
            return argv[++i];
        };
        if (arg == "--sizes") {
            o.sizes.clear();
            for (const std::string& item : split(value(), ',')) {
                const std::vector<std::string> wh = split(item, 'x');
                if (wh.size() > 2) throw std::invalid_argument("bad size: '" + item + "'");
                const std::size_t w = parse_count(wh[0], 1);
                o.sizes.push_back({w, wh.size() == 2 ? parse_count(wh[1], 1) : w});
            }
        } else if (arg == "--kernels") {
            o.kernels.clear();
            for (const std::string& name : split(value(), ',')) {
                const auto it = std::find_if(std::begin(kKernels), std::end(kKernels),
                                             [&](const KernelCase& k) { return k.name == name; });
                if (it == std::end(kKernels)) throw std::invalid_argument("unknown kernel: '" + name + "'");
                o.kernels.push_back(*it);
            }
        } else if (arg == "--methods") {
            o.methods.clear();
            for (const std::string& name : split(value(), ',')) {
                if (name == "direct") {
                    o.methods.push_back(lenia::ConvolutionMethod::Direct);
                } else if (name == "fft") {
                    o.methods.push_back(lenia::ConvolutionMethod::Fft);
                } else {
                    throw std::invalid_argument("unknown method: '" + name + "'");
                }
            }
        } else if (arg == "--steps") {
            o.steps = parse_count(value(), 1);
        } else if (arg == "--warmup") {
            o.warmup = parse_count(value(), 0);
        } else if (arg == "--samples") {
            o.samples = parse_count(value(), 1);
        } else if (arg == "--csv") {
            o.csv = true;
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    return o;
}

// Deterministic values in [0, 0.3): the mean neighborhood sits near Orbium's mu (0.15), so
// growth spans its range instead of every cell decaying at once.
lenia::Grid soup(std::size_t w, std::size_t h, std::uint32_t seed) {
    lenia::Grid g(w, h);
    for (float& c : g.cells) {
        seed = seed * 1664525u + 1013904223u;
        c = 0.3f * static_cast<float>(seed >> 8) / 16777216.0f;
    }
    return g;
}

bool same_bits(const lenia::Grid& a, const lenia::Grid& b) {
    return a.cells.size() == b.cells.size() &&
           std::memcmp(a.cells.data(), b.cells.data(), a.cells.size() * sizeof(float)) == 0;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct Verification {
    bool ok = false;
    std::string detail;
};

// Cells over kTol apart, and the largest difference.
std::size_t count_mismatches(const lenia::Grid& a, const lenia::Grid& b, float& max_diff) {
    max_diff = 0.0f;
    if (a.cells.size() != b.cells.size()) return a.cells.size() + b.cells.size();
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < a.cells.size(); ++i) {
        const float diff = std::fabs(a.cells[i] - b.cells[i]);
        if (diff > max_diff) max_diff = diff;
        if (!(diff <= kTol)) ++mismatches; // also catches NaN
    }
    return mismatches;
}

// With the stepper set to `method`: one generation on the GPU against the CPU step_lenia
// (tolerance kTol), at the case's size if the CPU can do it quickly, otherwise on a smaller
// grid (the shader is the same at any size); and two batched generations against two single GPU
// steps, which must match bit for bit. FFT is also checked against direct on the GPU at the
// case's size, first, so that call computes the kernel's spectrum (new kernel or size) and
// `transform_ms` receives its GPU time.
Verification verify(lenia::VulkanStepper& gpu, lenia::ConvolutionMethod method, Size size, const lenia::Grid& kernel,
                    const lenia::GrowthParams& params, std::optional<double>& transform_ms) {
    std::string detail;
    bool ok = true;
    char buf[256];
    gpu.set_convolution(method);
    if (method == lenia::ConvolutionMethod::Fft) {
        const lenia::Grid start = soup(size.w, size.h, 7u);
        const lenia::Grid fft = gpu.step_lenia(start, kernel, params);
        transform_ms = gpu.last_kernel_transform_ms();
        const bool ran_fft = gpu.last_convolution() == lenia::ConvolutionMethod::Fft;
        gpu.set_convolution(lenia::ConvolutionMethod::Direct);
        const lenia::Grid direct = gpu.step_lenia(start, kernel, params);
        gpu.set_convolution(method);
        float max_diff = 0.0f;
        const std::size_t mismatches = count_mismatches(fft, direct, max_diff);
        std::snprintf(buf, sizeof buf, "%s1 gen vs GPU direct at %zux%zu: max |fft-direct| %.2g, %zu over %.0e; ",
                      ran_fft ? "" : "FFT DID NOT RUN; ", size.w, size.h, static_cast<double>(max_diff), mismatches,
                      static_cast<double>(kTol));
        detail += buf;
        ok = ran_fft && mismatches == 0;
    }

    Size v = size;
    if (static_cast<double>(v.w) * v.h * kernel.cells.size() > kVerifyBudget) {
        v = {std::min(v.w, kVerifyFallback), std::min(v.h, kVerifyFallback)};
    }
    const lenia::Grid start = soup(v.w, v.h, 7u);
    const lenia::Grid cpu = lenia::step_lenia(start, kernel, params);
    const lenia::Grid one = gpu.step_lenia(start, kernel, params);
    float max_diff = 0.0f;
    const std::size_t mismatches = count_mismatches(one, cpu, max_diff);
    const bool batch_exact = same_bits(gpu.step_lenia(start, kernel, params, 2), gpu.step_lenia(one, kernel, params));

    std::snprintf(buf, sizeof buf, "1 gen vs CPU at %zux%zu: max |gpu-cpu| %.2g, %zu over %.0e; batch of 2 %s 2 single steps",
                  v.w, v.h, static_cast<double>(max_diff), mismatches, static_cast<double>(kTol),
                  batch_exact ? "==" : "!=");
    detail += buf;
    return {ok && mismatches == 0 && batch_exact, detail};
}

struct Result {
    std::string kernel;
    lenia::ConvolutionMethod method = lenia::ConvolutionMethod::Direct;
    bool unsupported = false; // FFT on a grid it does not handle: not run
    int radius = 0;
    std::size_t ksize = 0, taps = 0;
    Size grid;
    std::optional<double> transform_ms; // FFT kernel spectrum, once per kernel and grid size
    bool timed = false;
    double gpu_median = 0, gpu_min = 0, gpu_max = 0, wall_median = 0;
    bool deterministic = true;
    Verification check;
    std::string error;
};

std::string version_string(std::uint32_t v) {
    return std::to_string((v >> 22) & 0x7f) + "." + std::to_string((v >> 12) & 0x3ff) + "." +
           std::to_string(v & 0xfff);
}

// driverVersion is vendor-encoded; NVIDIA and Intel (Windows) differ from VK_MAKE_VERSION.
std::string driver_string(std::uint32_t vendor, std::uint32_t v) {
    std::string s;
    if (vendor == 0x10de) {
        s = std::to_string((v >> 22) & 0x3ff) + "." + std::to_string((v >> 14) & 0xff) + "." +
            std::to_string((v >> 6) & 0xff) + "." + std::to_string(v & 0x3f);
    } else if (vendor == 0x8086) {
        s = std::to_string(v >> 14) + "." + std::to_string(v & 0x3fff);
    } else {
        s = version_string(v);
    }
    char hex[16];
    std::snprintf(hex, sizeof hex, " (0x%08x)", v);
    return s + hex;
}

} // namespace

int main(int argc, char** argv) {
    Options opt;
    try {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
                usage(stdout);
                return 0;
            }
        }
        opt = parse_options(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lenia_bench: %s\n", e.what());
        usage(stderr);
        return 2;
    }

    lenia::Pattern orbium;
    try {
        orbium = lenia::load_pattern_file(std::string(LENIA_PATTERN_DIR) + "/orbium.lenia");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lenia_bench: cannot load the bundled Orbium: %s\n", e.what());
        return 1;
    }

    std::optional<lenia::VulkanStepper> stepper;
    try {
        stepper.emplace();
    } catch (const std::runtime_error& e) {
        std::printf("SKIP: %s\n", e.what());
        return kSkip;
    }
    lenia::VulkanStepper& gpu = *stepper;
    const lenia::VulkanDeviceInfo& dev = gpu.device_info();
    const lenia::GrowthParams& params = orbium.growth;

#ifdef NDEBUG
    const char* build = "Release";
#else
    const char* build = "Debug (CPU verification is slow; GPU timing is unaffected)";
#endif

    // Header: what was measured, on what. Prefixed with '#' in CSV mode.
    const char* hp = opt.csv ? "# " : "";
    char vendor[64];
    std::snprintf(vendor, sizeof vendor, "vendor 0x%04x, device 0x%04x", dev.vendor_id, dev.device_id);
    std::printf("%slenia_bench: batched Lenia generations, direct (shaders/lenia_step.comp) and/or FFT "
                "(shaders/fft_convolve.comp) convolution\n", hp);
    std::printf("%sdevice:     %s (%s, %s)\n", hp, dev.name.c_str(), dev.type.c_str(), vendor);
    std::printf("%sdriver:     %s, Vulkan %s\n", hp, driver_string(dev.vendor_id, dev.driver_version).c_str(),
                version_string(dev.api_version).c_str());
    if (dev.timestamps) {
        std::printf("%stimestamps: yes, %.3g ns per tick\n", hp, dev.timestamp_period_ns);
    } else {
        std::printf("%stimestamps: UNAVAILABLE on this compute queue; timing will be skipped\n", hp);
    }
    std::printf("%sgrowth:     %s, mu %g, sigma %g, dt %g (bundled Orbium); kernels: its core and rings, "
                "sum-normalized, radius varies\n",
                hp, params.function == lenia::GrowthFunction::Polynomial ? "polynomial" : "gaussian",
                static_cast<double>(params.mu), static_cast<double>(params.sigma), static_cast<double>(params.dt));
    std::printf("%smethod:     per case %zu warm-up + %zu timed batches of %zu generations (one submission "
                "each); GPU ms = timestamps around the dispatches; median reported\n",
                hp, opt.warmup, opt.samples, opt.steps);
    std::printf("%sfft:        %s\n", hp,
                dev.fft_max_size ? ("power-of-two sides " + std::to_string(lenia::kFftMinSize) + ".." +
                                    std::to_string(dev.fft_max_size) + "; kernel spectrum cached, its one-time "
                                    "cost reported apart (transform ms), not in the timed batches")
                                       .c_str()
                                 : "UNAVAILABLE on this device");
    std::printf("%sbuild:      %s\n", hp, build);

    std::vector<Result> results;
    const std::size_t total = opt.kernels.size() * opt.sizes.size() * opt.methods.size();
    bool all_ok = true;
    bool tdr_warning = false;
    for (const KernelCase& kc : opt.kernels) {
        lenia::KernelShape shape = orbium.kernel;
        if (kc.radius > 0) shape.radius = kc.radius;
        const lenia::Grid kernel =
            lenia::normalize_kernel(lenia::make_kernel(shape), lenia::KernelNormalization::Sum);

        // Methods side by side per case, from the same soup, with the same batches.
        for (const Size size : opt.sizes) for (const lenia::ConvolutionMethod method : opt.methods) {
            Result r;
            r.kernel = kc.name;
            r.method = method;
            r.radius = shape.radius;
            r.ksize = kernel.width;
            r.taps = lenia::kernel_stats(kernel).nonzero;
            r.grid = size;
            std::fprintf(stderr, "[%zu/%zu] %s R%d %zux%zu kernel @ %zux%zu grid, %s: ", results.size() + 1, total,
                         r.kernel.c_str(), r.radius, kernel.width, kernel.height, size.w, size.h, method_name(method));
            if (method == lenia::ConvolutionMethod::Fft && !gpu.fft_supported(size.w, size.h)) {
                r.unsupported = true;
                std::fprintf(stderr, "not supported for this grid (direct would run instead); skipped\n");
                results.push_back(std::move(r));
                continue;
            }
            try {
                r.check = verify(gpu, method, size, kernel, params, r.transform_ms);
                std::fprintf(stderr, "verify %s (%s)", r.check.ok ? "ok" : "FAILED", r.check.detail.c_str());
                if (r.transform_ms) std::fprintf(stderr, "; kernel transform %.3f ms", *r.transform_ms);

                if (dev.timestamps) {
                    const lenia::Grid start = soup(size.w, size.h, 1u);
                    for (std::size_t i = 0; i < opt.warmup; ++i) gpu.step_lenia(start, kernel, params, opt.steps);
                    std::vector<double> gpu_ms, wall_ms;
                    lenia::Grid first;
                    for (std::size_t i = 0; i < opt.samples; ++i) {
                        const auto t0 = std::chrono::steady_clock::now();
                        lenia::Grid out = gpu.step_lenia(start, kernel, params, opt.steps);
                        const auto t1 = std::chrono::steady_clock::now();
                        const std::optional<double> ms = gpu.last_gpu_ms();
                        if (!ms) throw std::runtime_error("no GPU timestamp for a batch");
                        gpu_ms.push_back(*ms);
                        wall_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                        // Same input, same device: every sample must give the same cells.
                        if (i == 0) {
                            first = std::move(out);
                        } else if (!same_bits(out, first)) {
                            r.deterministic = false;
                        }
                    }
                    r.timed = true;
                    r.gpu_median = median(gpu_ms);
                    r.gpu_min = *std::min_element(gpu_ms.begin(), gpu_ms.end());
                    r.gpu_max = *std::max_element(gpu_ms.begin(), gpu_ms.end());
                    r.wall_median = median(wall_ms);
                    if (r.gpu_max > kTdrWarnMs) tdr_warning = true;
                    std::fprintf(stderr, "; median %.3f ms per batch%s", r.gpu_median,
                                 r.deterministic ? "" : "; samples DIFFER");
                }
            } catch (const std::exception& e) {
                r.error = e.what();
                std::fprintf(stderr, "ERROR: %s", e.what());
            }
            std::fprintf(stderr, "\n");
            all_ok = all_ok && r.error.empty() && r.check.ok && r.deterministic;
            results.push_back(std::move(r));
        }
    }
    gpu.set_convolution(lenia::ConvolutionMethod::Direct);

    auto verdict = [](const Result& r) -> std::string {
        if (r.unsupported) return "n/a: grid not supported by fft";
        if (!r.error.empty()) return "ERROR: " + r.error;
        if (!r.check.ok) return "FAILED";
        return r.deterministic ? "ok" : "NONDETERMINISTIC";
    };

    if (opt.csv) {
        std::printf("kernel,method,radius,kernel_size,taps,width,height,steps,warmup,samples,gpu_ms_per_gen_median,"
                    "gpu_ms_per_batch_median,gpu_ms_per_batch_min,gpu_ms_per_batch_max,mcells_per_s,"
                    "wall_ms_per_batch_median,kernel_transform_ms,verify\n");
        for (const Result& r : results) {
            std::printf("%s,%s,%d,%zu,%zu,%zu,%zu,%zu,%zu,%zu,", r.kernel.c_str(), method_name(r.method), r.radius,
                        r.ksize, r.taps, r.grid.w, r.grid.h, opt.steps, opt.warmup, opt.samples);
            if (r.timed) {
                const double mcells = static_cast<double>(r.grid.w) * r.grid.h * opt.steps / (r.gpu_median * 1e3);
                std::printf("%.6f,%.6f,%.6f,%.6f,%.1f,%.3f,", r.gpu_median / opt.steps, r.gpu_median, r.gpu_min,
                            r.gpu_max, mcells, r.wall_median);
            } else {
                std::printf(",,,,,,");
            }
            if (r.transform_ms) std::printf("%.6f", *r.transform_ms);
            std::printf(",\"%s\"\n", verdict(r).c_str());
        }
    } else {
        std::printf("\n%-7s %-6s %4s %7s %5s %11s %10s %12s %10s %10s %10s %9s %9s  %s\n", "kernel", "method", "R",
                    "size", "taps", "grid", "ms/gen", "ms/batch med", "min", "max", "Mcell/s", "wall ms",
                    "xform ms", "verify");
        for (const Result& r : results) {
            char ksize[16], grid[24], xform[16];
            std::snprintf(ksize, sizeof ksize, "%zux%zu", r.ksize, r.ksize);
            std::snprintf(grid, sizeof grid, "%zux%zu", r.grid.w, r.grid.h);
            if (r.transform_ms) {
                std::snprintf(xform, sizeof xform, "%.3f", *r.transform_ms);
            } else {
                std::snprintf(xform, sizeof xform, "-");
            }
            std::printf("%-7s %-6s %4d %7s %5zu %11s ", r.kernel.c_str(), method_name(r.method), r.radius, ksize, r.taps,
                        grid);
            if (r.timed) {
                const double mcells = static_cast<double>(r.grid.w) * r.grid.h * opt.steps / (r.gpu_median * 1e3);
                std::printf("%10.4f %12.3f %10.3f %10.3f %10.0f %9.3f", r.gpu_median / opt.steps, r.gpu_median,
                            r.gpu_min, r.gpu_max, mcells, r.wall_median);
            } else {
                std::printf("%10s %12s %10s %10s %10s %9s", "skipped", "-", "-", "-", "-", "-");
            }
            std::printf(" %9s  %s\n", xform, verdict(r).c_str());
        }
        std::printf("\nms = GPU time from timestamp queries (median of %zu batches of %zu generations); "
                    "wall ms also includes upload, submission and readback. xform ms = the FFT kernel spectrum, "
                    "computed once per kernel and grid size, outside the timed batches.\n",
                    opt.samples, opt.steps);
    }

    if (tdr_warning) {
        std::fprintf(stderr, "WARNING: a batch took over %.0f ms; near the Windows GPU timeout (TDR). "
                             "Lower --steps for large cases.\n", kTdrWarnMs);
    }
    if (!all_ok) {
        std::fprintf(stderr, "lenia_bench: FAILED (see verify column)\n");
        return 1;
    }
    if (!dev.timestamps) {
        std::fprintf(stderr, "lenia_bench: SKIPPED timing: the device has no timestamp queries on its compute "
                             "queue (verification passed)\n");
        return kSkip;
    }
    return 0;
}
