#include "lenia/fft_convolve.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace lenia {

namespace {

bool is_pow2(std::size_t n) { return n != 0 && (n & (n - 1)) == 0; }

// The shader's vec2 arithmetic, in the same order.
struct Complex {
    float x = 0.0f, y = 0.0f;
};
Complex operator+(Complex a, Complex b) { return {a.x + b.x, a.y + b.y}; }
Complex operator-(Complex a, Complex b) { return {a.x - b.x, a.y - b.y}; }
Complex operator*(float s, Complex a) { return {s * a.x, s * a.y}; }
Complex cmul(Complex a, Complex b) { return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }

std::uint32_t bitrev(std::uint32_t i, std::uint32_t bits) {
    std::uint32_t r = 0;
    for (std::uint32_t b = 0; b < bits; ++b) r |= ((i >> b) & 1u) << (bits - 1 - b);
    return r;
}

// One line's transforms; `buf` is the shader's shared array.
struct Line {
    std::vector<Complex>& buf;
    const std::vector<Complex>& tw;
    std::uint32_t log2_tw;

    // Forward, natural order in, bit-reversed out (decimation in frequency).
    void dif(std::uint32_t n_log2) {
        const std::uint32_t half_n = 1u << (n_log2 - 1u);
        for (std::uint32_t s = n_log2; s >= 1u; --s) {
            const std::uint32_t h = 1u << (s - 1u);
            for (std::uint32_t b = 0; b < half_n; ++b) {
                const std::uint32_t pos = b & (h - 1u);
                const std::uint32_t i0 = ((b >> (s - 1u)) << s) + pos;
                const std::uint32_t i1 = i0 + h;
                const Complex u = buf[i0], v = buf[i1];
                buf[i0] = u + v;
                buf[i1] = cmul(u - v, tw[pos << (log2_tw - s)]);
            }
        }
    }

    // Inverse, unnormalized, bit-reversed order in, natural out (decimation in time).
    void dit_inverse(std::uint32_t n_log2) {
        const std::uint32_t half_n = 1u << (n_log2 - 1u);
        for (std::uint32_t s = 1u; s <= n_log2; ++s) {
            const std::uint32_t h = 1u << (s - 1u);
            for (std::uint32_t b = 0; b < half_n; ++b) {
                const std::uint32_t pos = b & (h - 1u);
                const std::uint32_t i0 = ((b >> (s - 1u)) << s) + pos;
                const std::uint32_t i1 = i0 + h;
                Complex w = tw[pos << (log2_tw - s)];
                w.y = -w.y;
                const Complex u = buf[i0], v = cmul(buf[i1], w);
                buf[i0] = u + v;
                buf[i1] = u - v;
            }
        }
    }
};

void validate(const Grid& g, const char* name) {
    if (g.cells.size() != g.width * g.height) {
        throw std::invalid_argument(std::string(name) + " cell count does not match width * height");
    }
}

} // namespace

std::size_t fft_max_size(std::size_t shared_bytes) {
    std::size_t n = kFftMaxSize;
    while (n >= kFftMinSize && n * 8 > shared_bytes) n /= 2;
    return n >= kFftMinSize ? n : 0;
}

bool fft_size_supported(std::size_t width, std::size_t height, std::size_t max_size) {
    const auto ok = [&](std::size_t n) { return is_pow2(n) && n >= kFftMinSize && n <= max_size && n <= kFftMaxSize; };
    return ok(width) && ok(height);
}

std::uint32_t fft_log2(std::size_t n) {
    std::uint32_t l = 0;
    while ((std::size_t{1} << l) < n) ++l;
    return l;
}

std::vector<float> fft_twiddles(std::size_t n) {
    if (!is_pow2(n) || n < 2) throw std::invalid_argument("fft_twiddles: n must be a power of two >= 2");
    const double two_pi = 6.283185307179586476925286766559;
    std::vector<float> t(n); // n / 2 complex values
    for (std::size_t i = 0; i < n / 2; ++i) {
        const double a = two_pi * static_cast<double>(i) / static_cast<double>(n);
        t[2 * i] = static_cast<float>(std::cos(a));
        t[2 * i + 1] = static_cast<float>(-std::sin(a));
    }
    return t;
}

std::size_t fft_spectrum_size(std::size_t width, std::size_t height) { return (width / 2 + 1) * height; }

std::uint32_t fft_pass_groups(FftPass pass, std::size_t width, std::size_t height) {
    switch (pass) {
    case FftPass::KernelColumns:
    case FftPass::ConvolveColumns: return static_cast<std::uint32_t>(width / 2 + 1);
    default: return static_cast<std::uint32_t>(height / 2);
    }
}

Grid convolve_fft(const Grid& grid, const Grid& kernel) {
    if (kernel.width % 2 == 0 || kernel.height % 2 == 0) {
        throw std::invalid_argument("kernel dimensions must be odd");
    }
    validate(grid, "grid");
    validate(kernel, "kernel");
    if (!fft_size_supported(grid.width, grid.height)) {
        throw std::invalid_argument("convolve_fft: grid sides must be powers of two in [8, 8192]");
    }

    const std::uint32_t W = static_cast<std::uint32_t>(grid.width);
    const std::uint32_t H = static_cast<std::uint32_t>(grid.height);
    const std::uint32_t lw = fft_log2(W), lh = fft_log2(H);
    const std::uint32_t wh = W / 2u + 1u;
    const std::size_t n_tw = W > H ? W : H;
    const std::uint32_t log2_tw = fft_log2(n_tw);
    std::vector<Complex> tw(n_tw / 2);
    {
        const std::vector<float> t = fft_twiddles(n_tw);
        for (std::size_t i = 0; i < tw.size(); ++i) tw[i] = {t[2 * i], t[2 * i + 1]};
    }
    std::vector<Complex> buf(n_tw), spec(fft_spectrum_size(W, H)), khat(spec.size());
    Line line{buf, tw, log2_tw};

    // Kernel tap (kx, ky) lands on torus cell ((kx - cx) mod W, (ky - cy) mod H), as in convolve;
    // taps landing on the same cell (kernels larger than the grid) add up.
    const std::uint32_t kw = static_cast<std::uint32_t>(kernel.width), kh = static_cast<std::uint32_t>(kernel.height);
    const auto kernel_folded = [&](std::uint32_t x, std::uint32_t y) {
        float sum = 0.0f;
        for (std::uint32_t ky = (y + kh / 2u) % H; ky < kh; ky += H) {
            for (std::uint32_t kx = (x + kw / 2u) % W; kx < kw; kx += W) sum += kernel.cells[ky * kw + kx];
        }
        return sum;
    };

    // Rows 2j and 2j + 1 as the real and imaginary parts of one transform, separated into the
    // two rows' half spectra: A(k) = (Z(k) + conj Z(-k)) / 2, B(k) = (Z(k) - conj Z(-k)) / 2i.
    const auto rows_forward = [&](bool from_kernel) {
        for (std::uint32_t j = 0; j < H / 2u; ++j) {
            const std::uint32_t y0 = 2u * j, y1 = y0 + 1u;
            for (std::uint32_t x = 0; x < W; ++x) {
                buf[x] = from_kernel ? Complex{kernel_folded(x, y0), kernel_folded(x, y1)}
                                     : Complex{grid.cells[y0 * W + x], grid.cells[y1 * W + x]};
            }
            line.dif(lw);
            for (std::uint32_t k = 0; k < wh; ++k) {
                const Complex zk = buf[bitrev(k, lw)];
                Complex zm = buf[bitrev((W - k) & (W - 1u), lw)];
                zm.y = -zm.y;
                const Complex d = zk - zm;
                spec[y0 * wh + k] = 0.5f * (zk + zm);
                spec[y1 * wh + k] = 0.5f * Complex{d.y, -d.x};
            }
        }
    };

    // Column k of the half spectrum, forward (bit-reversed order); khat is kept in that order.
    const auto columns = [&](bool kernel_pass) {
        const float scale = 1.0f / (static_cast<float>(W) * static_cast<float>(H));
        for (std::uint32_t k = 0; k < wh; ++k) {
            for (std::uint32_t y = 0; y < H; ++y) buf[y] = spec[y * wh + k];
            line.dif(lh);
            if (kernel_pass) {
                for (std::uint32_t i = 0; i < H; ++i) khat[k * H + i] = scale * buf[i];
                continue;
            }
            for (std::uint32_t i = 0; i < H; ++i) buf[i] = cmul(buf[i], khat[k * H + i]);
            line.dit_inverse(lh);
            for (std::uint32_t y = 0; y < H; ++y) spec[y * wh + k] = buf[y];
        }
    };

    rows_forward(true);
    columns(true);
    rows_forward(false);
    columns(false);

    // Both rows' spectra back as one transform: Z(k) = P(k) + i Q(k), P and Q Hermitian.
    Grid out(W, H, 0.0f);
    for (std::uint32_t j = 0; j < H / 2u; ++j) {
        const std::uint32_t y0 = 2u * j, y1 = y0 + 1u;
        for (std::uint32_t k = 0; k < W; ++k) {
            Complex z;
            if (k <= W / 2u) {
                const Complex P = spec[y0 * wh + k], Q = spec[y1 * wh + k];
                z = {P.x - Q.y, P.y + Q.x};
            } else {
                const Complex P = spec[y0 * wh + (W - k)], Q = spec[y1 * wh + (W - k)];
                z = {P.x + Q.y, Q.x - P.y};
            }
            buf[bitrev(k, lw)] = z;
        }
        line.dit_inverse(lw);
        for (std::uint32_t x = 0; x < W; ++x) {
            out.cells[y0 * W + x] = buf[x].x;
            out.cells[y1 * W + x] = buf[x].y;
        }
    }
    return out;
}

} // namespace lenia
