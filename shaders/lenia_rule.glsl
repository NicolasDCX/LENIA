// Per-cell rule interpreter shared by workbench_step.comp and rule_step.comp. Mirrors
// lenia::run_rule (rule.hpp, src/rule.cpp) op for op; the program comes from lenia::pack_rule.
//
// The including shader defines LENIA_RULE_BINDING (the program's binding in set 0) and
// includes lenia_math.glsl first.

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
    uint rule_pad;
    RuleOp rule_ops[];
};

float lenia_finite_or_zero(float v) { return (isnan(v) || isinf(v)) ? 0.0 : v; }

// Next state of cell (x, y) with density `a` and neighborhood `u`; `growth_display` receives
// the Growth display value (the program's growth register, else next - a).
float lenia_rule(uint x, uint y, uint width, uint height, float a, float u, out float growth_display) {
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
        default: break;
        }
        r[i] = v;
    }
    const float next = lenia_finite_or_zero(rule_output < count ? r[rule_output] : a);
    growth_display = lenia_finite_or_zero(rule_growth < count ? r[rule_growth] : next - a);
    return next;
}
