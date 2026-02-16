#version 300 es
// Fullscreen triangle: 3 vertices, no VBO needed.
out vec2 v_uv;
void main() {
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  v_uv = vec2((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
  gl_Position = vec4(x, y, 0.0, 1.0);
}
