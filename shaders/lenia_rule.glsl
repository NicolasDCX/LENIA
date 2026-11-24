// Per-cell rule interpreter shared by workbench_step.comp and rule_step.comp. Mirrors
// lenia::run_rule (rule.hpp, src/rule.cpp) op for op; the program comes from lenia::pack_rule.
//
// The including shader defines LENIA_RULE_BINDING (the program's binding in set 0) and
// includes lenia_math.glsl first. Convolve ops read their kernels from `ker` (at the offset
// pack_rule gave them, after the active kernel) and their field from `src`: the density in the
// first width * height floats, the nutrient in the next when `planes` is 2.

#define LENIA_RULE_MAX_OPS 64u // lenia::kMaxRuleOps
#define LENIA_NO_REGISTER 0xffffffffu

struct RuleOp {
    uint op;
    uint a;
    uint b;
    uint c;
    uint m0;
    uint m1;
    float k0;
    float k1;
    float k2;
    float k3;
    float k4;
    float k5;
};

layout(std430, set = 0, binding = LENIA_RULE_BINDING) readonly buffer Rule {
    uint rule_count;
    uint rule_output;
    uint rule_growth;
    uint rule_nutrient;
    RuleOp rule_ops[];
};

float lenia_finite_or_zero(float v) { return (isnan(v) || isinf(v)) ? 0.0 : v; }

// K ∗ field at (x, y), as lenia_neighborhood: the kw x kh kernel at ker[koff], the field at
// src[base] (lenia::convolve, toroidal).
float lenia_rule_convolve(uint x, uint y, uint width, uint height, uint koff, uint kw, uint kh, uint base) {
    const uint cx = kw / 2u;
    const uint cy = kh / 2u;
    const uint bx = x + cx + width * (cx / width + 1u);
    const uint by = y + cy + height * (cy / height + 1u);
    float sum = 0.0;
    for (uint ky = 0u; ky < kh; ++ky) {
        const uint row = base + ((by - ky) % height) * width;
        for (uint kx = 0u; kx < kw; ++kx) {
            const float w = ker[koff + ky * kw + kx];
            if (w == 0.0) continue;
            sum += w * src[row + (bx - kx) % width];
        }
    }
    return sum;
}

// Next state of cell (x, y) with density `a`, nutrient `n` and neighborhood `u`; `planes` is 2
// when `src` holds the nutrient after the density (else Convolve ops on it read 0).
// `growth_display` receives the Growth display value (the program's growth register, else
// next - a), `next_n` the next nutrient (the program's nutrient register, else n) and
// `preview_value` register `preview` (0 if it is not one).
float lenia_rule(uint x, uint y, uint width, uint height, uint planes, float a, float n, float u, uint preview,
                 out float growth_display, out float next_n, out float preview_value) {
    float r[LENIA_RULE_MAX_OPS];
    const uint count = min(rule_count, LENIA_RULE_MAX_OPS);
    for (uint i = 0u; i < count; ++i) {
        const RuleOp o = rule_ops[i];
        // compile_rule only reads earlier registers; the mask keeps a bad index in bounds.
        const float ra = r[o.a & (LENIA_RULE_MAX_OPS - 1u)];
        const float rb = r[o.b & (LENIA_RULE_MAX_OPS - 1u)];
        const float rc = r[o.c & (LENIA_RULE_MAX_OPS - 1u)];
        float v = 0.0;
        switch (o.op) {
        case 0u: v = o.k0; break;                                        // Const
        case 1u: v = a; break;                                           // Density
        case 2u: v = u; break;                                           // Neighborhood
        case 3u: v = lenia_growth(ra, rb, rc, o.m0 == 1u ? 1u : 0u); break; // Growth
        case 4u: v = ra + rb; break;                                     // Add
        case 5u: v = ra * rb; break;                                     // Multiply
        case 6u: v = min(max(ra, rb), rc); break;                        // Clamp
        case 7u: {                                                       // Compare
            bool t = false;
            if (o.m0 == 0u) t = ra > rb;
            else if (o.m0 == 1u) t = ra >= rb;
            else if (o.m0 == 2u) t = ra < rb;
            else if (o.m0 == 3u) t = ra <= rb;
            else if (o.m0 == 4u) t = ra == rb;
            else if (o.m0 == 5u) t = ra != rb;
            v = t ? 1.0 : 0.0;
            break;
        }
        case 8u: v = ra >= 0.5 ? rb : rc; break;                         // Select
        case 9u: {                                                       // Life
            const float n = floor(rb + 0.5);
            const uint mask = ra >= 0.5 ? o.m1 : o.m0;
            v = (n >= 0.0 && n <= 8.0) ? float((mask >> uint(n)) & 1u) : 0.0;
            break;
        }
        case 10u:                                                        // EnvField
            v = o.k4 * lenia_environment(o.m0, x, y, width, height, o.k0, o.k1, o.k2, o.k3);
            break;
        case 11u: v = n; break;                                          // Nutrient
        case 12u:                                                        // Convolve
            if (o.m1 == 0u || planes > 1u) {
                v = lenia_rule_convolve(x, y, width, height, o.a, o.b, o.c, o.m1 == 0u ? 0u : width * height);
            }
            break;
        default: break;
        }
        r[i] = v;
    }
    const float next = lenia_finite_or_zero(rule_output < count ? r[rule_output] : a);
    growth_display = lenia_finite_or_zero(rule_growth < count ? r[rule_growth] : next - a);
    next_n = lenia_finite_or_zero(rule_nutrient < count ? r[rule_nutrient] : n);
    preview_value = preview < count ? r[preview] : 0.0;
    return next;
}
