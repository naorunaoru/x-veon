// Compute shader: bins a 2D rgba8unorm texture into 4 × 256 histogram bins.
// Layout: bins[0..255] = R, bins[256..511] = G, bins[512..767] = B, bins[768..1023] = L (Rec.709)

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 1024>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(t_input);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let c = textureLoad(t_input, vec2i(gid.xy), 0);

    let ri = u32(clamp(c.r * 255.0 + 0.5, 0.0, 255.0));
    let gi = u32(clamp(c.g * 255.0 + 0.5, 0.0, 255.0));
    let bi = u32(clamp(c.b * 255.0 + 0.5, 0.0, 255.0));
    let li = u32(clamp((0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b) * 255.0 + 0.5, 0.0, 255.0));

    atomicAdd(&bins[ri], 1u);
    atomicAdd(&bins[256u + gi], 1u);
    atomicAdd(&bins[512u + bi], 1u);
    atomicAdd(&bins[768u + li], 1u);
}
