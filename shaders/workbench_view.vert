#version 450

// Fullscreen triangle; workbench_view.frag maps gl_FragCoord to grid cells.
void main() {
    const vec2 pos = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
