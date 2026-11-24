// Lenia math shared by lenia_step.comp and workbench_step.comp. Mirrors lenia::convolve and
// lenia::step_lenia:
//
//   U      = sum over (dx, dy) of kernel(dx, dy) * grid(x - dx, y - dy)   (toroidal wrap)
//   z      = (U - mu) / sigma
//   growth = 2 * exp(-z^2 / 2) - 1           (function 0, lenia::GrowthFunction::Gaussian)
//            2 * max(0, 1 - z^2 / 9)^4 - 1   (function 1, lenia::GrowthFunction::Polynomial)
//   next   = clamp(current + dt * growth, 0, 1)
//
// The including shader must declare float arrays `src` (grid) and `ker` (kernel) first.

float lenia_neighborhood(uint x, uint y, uint width, uint height, uint kwidth, uint kheight) {
    const uint cx = kwidth / 2u;
    const uint cy = kheight / 2u;

    // Source column for kernel column kx is x - (kx - cx). GLSL leaves % on negative ints
    // undefined, so add a multiple of the grid size larger than cx (cy) to stay unsigned.
    // This also handles kernels larger than the grid.
    const uint bx = x + cx + width * (cx / width + 1u);
    const uint by = y + cy + height * (cy / height + 1u);

    float sum = 0.0;
    for (uint ky = 0u; ky < kheight; ++ky) {
        const uint row = ((by - ky) % height) * width;
        for (uint kx = 0u; kx < kwidth; ++kx) {
            const float w = ker[ky * kwidth + kx];
            if (w == 0.0) continue;
            sum += w * src[row + (bx - kx) % width];
        }
    }
    return sum;
}

float lenia_growth(float u, float mu, float sigma, uint kind) {
    const float z = (u - mu) / sigma;
    if (kind == 1u) {
        const float q = max(0.0, 1.0 - z * z / 9.0);
        return 2.0 * (q * q) * (q * q) - 1.0;
    }
    return 2.0 * exp(-0.5 * z * z) - 1.0;
}

// Environment field E(x, y) in [-1, 1]; mirrors lenia::environment_field (environment.hpp),
// `pattern` is a lenia::EnvironmentPattern, c and s are cos and sin of the angle.
float lenia_environment(uint pattern, uint x, uint y, uint width, uint height, float scale, float c, float s,
                        float offset) {
    const float PI = 3.14159265358979;
    const float dx = float(x) - float(width / 2u);
    const float dy = float(y) - float(height / 2u);
    scale = max(scale, 1.0);
    const float u = dx * c + dy * s - offset;
    if (pattern == 0u) return 1.0;
    if (pattern == 1u) {
        const vec2 e = vec2(dx - offset * c, dy - offset * s);
        return cos(PI * min(length(e) / scale, 1.0));
    }
    if (pattern == 2u) return clamp(u / scale, -1.0, 1.0);
    if (pattern == 3u) return sin(2.0 * PI * u / scale);
    if (pattern == 4u) return cos(2.0 * PI * (length(vec2(dx, dy)) - offset) / scale);
    return 0.0;
}

float lenia_next(float current, float growth, float dt) {
    return clamp(current + dt * growth, 0.0, 1.0);
}
