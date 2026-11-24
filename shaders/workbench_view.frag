#version 450

// Draws the simulation straight from its device buffers. Nearest-cell sampling; row 0 of the
// grid is the top of the image.
//
// With program == 0 one field goes through a palette lookup table (the Simulation panel's
// display). With program == 1 the display graph's program (lenia::pack_display) colors each
// cell from its density, neighborhood and growth; display_color mirrors lenia::run_display
// (src/display_graph.cpp) op for op.

layout(std430, set = 0, binding = 0) readonly buffer Density { float density[]; };
layout(std430, set = 0, binding = 1) readonly buffer Neighborhood { float neighborhood[]; };
layout(std430, set = 0, binding = 2) readonly buffer Growth { float growth_field[]; };
layout(std430, set = 0, binding = 3) readonly buffer Palettes { vec4 palettes[]; }; // 256 entries each

#define DISPLAY_MAX_OPS 32u    // lenia::kMaxDisplayOps
#define DISPLAY_MAX_STOPS 64u  // lenia::kMaxDisplayStops

struct DisplayOp {
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

layout(std430, set = 0, binding = 4) readonly buffer Display {
    uint display_count;
    uint display_output;
    uint display_stop_count;
    uint display_pad;
    DisplayOp display_ops[DISPLAY_MAX_OPS];
    vec4 display_stops[DISPLAY_MAX_STOPS]; // position, r, g, b
};

layout(push_constant) uniform View {
    vec2 origin;          // framebuffer pixel of the grid's top-left corner
    float cells_per_pixel;
    uint width;
    uint height;
    uint field;           // 0 density, 1 neighborhood, 2 growth
    uint palette;         // index of the 256-entry table
    float lo;             // field value mapped to the first palette entry
    float hi;             // field value mapped to the last palette entry
    uint linear_target;   // 1 if the swapchain is sRGB, so palette colors must be linearized
    uint program;         // 1: the Display program colors the cells
} v;

layout(location = 0) out vec4 color;

// Clamp to [0, 1] with NaN (and negatives) giving 0, as lenia's unit().
float unit(float t) {
    if (!(t >= 0.0)) return 0.0;
    return t > 1.0 ? 1.0 : t;
}

vec3 display_ramp(uint first, uint n, float t) {
    if (n == 0u) return vec3(0.0);
    vec4 prev = display_stops[first];
    if (!(t > prev.x)) return prev.yzw;
    for (uint j = 1u; j < n; ++j) {
        const vec4 s = display_stops[first + j];
        if (t <= s.x) {
            const float d = s.x - prev.x;
            const float f = d > 0.0 ? (t - prev.x) / d : 1.0;
            return prev.yzw + (s.yzw - prev.yzw) * f;
        }
        prev = s;
    }
    return prev.yzw;
}

vec3 display_color(uint i) {
    vec4 r[DISPLAY_MAX_OPS];
    const uint count = min(display_count, DISPLAY_MAX_OPS);
    const uint stop_count = min(display_stop_count, DISPLAY_MAX_STOPS);
    for (uint k = 0u; k < count; ++k) {
        const DisplayOp o = display_ops[k];
        // compile_display only reads earlier registers; the mask keeps a bad index in bounds.
        const vec4 ra = r[o.a & (DISPLAY_MAX_OPS - 1u)];
        const vec4 rb = r[o.b & (DISPLAY_MAX_OPS - 1u)];
        const vec4 rc = r[o.c & (DISPLAY_MAX_OPS - 1u)];
        vec4 val = vec4(0.0);
        switch (o.op) {
        case 0u: val = vec4(o.k0, o.k1, o.k2, o.k3); break; // Const
        case 1u: {                                           // Field
            const float f = o.m0 == 0u ? density[i] : (o.m0 == 1u ? neighborhood[i] : growth_field[i]);
            val = vec4(f, f, f, 1.0);
            break;
        }
        case 2u: {                                           // Ramp
            const float span = o.k1 - o.k0;
            const float t = unit(span != 0.0 ? (ra.x - o.k0) / span : 0.0);
            const uint first = min(o.m0, stop_count);
            const uint n = min(o.m1, stop_count - first);
            val = vec4(display_ramp(first, n, t), 1.0);
            break;
        }
        case 3u: {                                           // Mix
            const float f = unit(rc.x);
            vec3 x = vec3(0.0);
            if (o.m0 == 0u) x = ra.rgb + (rb.rgb - ra.rgb) * f;
            else if (o.m0 == 1u) x = ra.rgb + rb.rgb * f;
            else if (o.m0 == 2u) x = ra.rgb * (vec3(1.0) + (rb.rgb - vec3(1.0)) * f);
            else if (o.m0 == 3u) x = vec3(1.0) - (vec3(1.0) - ra.rgb) * (vec3(1.0) - rb.rgb * f);
            val = vec4(unit(x.r), unit(x.g), unit(x.b), 1.0);
            break;
        }
        default: break;
        }
        r[k] = val;
    }
    if (display_output >= count) return vec3(0.0);
    const vec4 o = r[display_output];
    return vec3(unit(o.r), unit(o.g), unit(o.b));
}

void main() {
    const vec2 g = (gl_FragCoord.xy - v.origin) * v.cells_per_pixel;
    if (g.x < 0.0 || g.y < 0.0 || g.x >= float(v.width) || g.y >= float(v.height)) {
        color = vec4(0.06, 0.06, 0.07, 1.0);
        if (v.linear_target != 0u) color.rgb = pow(color.rgb, vec3(2.2));
        return;
    }
    const uint i = min(uint(g.y), v.height - 1u) * v.width + min(uint(g.x), v.width - 1u);
    if (v.program != 0u) {
        color = vec4(display_color(i), 1.0);
    } else {
        const float value = v.field == 0u ? density[i] : (v.field == 1u ? neighborhood[i] : growth_field[i]);
        const float span = v.hi - v.lo;
        const float t = clamp(span != 0.0 ? (value - v.lo) / span : 0.0, 0.0, 1.0);
        color = palettes[v.palette * 256u + uint(t * 255.0 + 0.5)];
    }
    if (v.linear_target != 0u) color.rgb = pow(color.rgb, vec3(2.2));
}
