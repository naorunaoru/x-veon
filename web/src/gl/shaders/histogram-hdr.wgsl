// Compute shader: bins an rgba32float texture into 4 × 256 histogram bins.
// Supports two modes: linear [0, max] and log2 [ev_min, ev_max].
// Stride is configurable (4 for full-res imageTex, 1 for downsampled display tex).
// Layout: bins[0..255] = R, bins[256..511] = G, bins[512..767] = B, bins[768..1023] = L (Rec.709)

struct Params {
  exposure: f32,
  wb_temp: f32,
  wb_tint: f32,
  mode: f32,       // 0 = linear, 1 = log2
  range_lo: f32,   // linear: 0, log2: ev_min (-8)
  range_hi: f32,   // linear: max (2.0), log2: ev_max (8)
  stride: f32,     // subsampling stride (cast to u32)
};

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 1024>;
@group(0) @binding(2) var<uniform> params: Params;

fn to_bin(value: f32) -> u32 {
    var t: f32;
    if (params.mode > 0.5) {
        let ev = log2(max(value, 1e-10));
        t = (ev - params.range_lo) / (params.range_hi - params.range_lo);
    } else {
        t = (value - params.range_lo) / (params.range_hi - params.range_lo);
    }
    return u32(clamp(t * 256.0, 0.0, 255.0));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(t_input);
    let s = u32(params.stride);
    let coord = gid.xy * s;
    if (coord.x >= dims.x || coord.y >= dims.y) { return; }

    var c = textureLoad(t_input, vec2i(coord), 0).rgb;

    // Exposure + white balance (matching opendrt.wgsl preprocessing).
    // For display-mode histogram, these are set to 0 (pass-through).
    c *= exp2(params.exposure);
    c.x *= exp2(params.wb_temp);
    c.y *= exp2(-params.wb_tint);
    c.z *= exp2(-params.wb_temp);
    c /= 0.2126 * exp2(params.wb_temp) + 0.7152 * exp2(-params.wb_tint) + 0.0722 * exp2(-params.wb_temp);

    let lum = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;

    atomicAdd(&bins[to_bin(c.x)], 1u);
    atomicAdd(&bins[256u + to_bin(c.y)], 1u);
    atomicAdd(&bins[512u + to_bin(c.z)], 1u);
    atomicAdd(&bins[768u + to_bin(lum)], 1u);
}
