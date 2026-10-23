// Compute shader: bins an rgba32float texture into 4 × 256 histogram bins.
// Supports two modes: linear [0, max] and log2 [ev_min, ev_max].
// Stride is configurable (4 for full-res imageTex, 1 for downsampled display tex).
// Layout: bins[0..255] = R, bins[256..511] = G, bins[512..767] = B, bins[768..1023] = L (Rec.709)
// The scene pass can apply the lens vignetting gain (the geometry is ignored): each texel is a source pixel,
// so its gain is the table's .w at the texel's radius from the optical centre. The display pass never does;
// its source is already corrected.

struct Params {
  exposure: f32,
  wb_temp: f32,
  wb_tint: f32,
  mode: f32,       // 0 = linear, 1 = log2
  range_lo: f32,   // linear: 0, log2: ev_min (-8)
  range_hi: f32,   // linear: max (2.0), log2: ev_max (8)
  stride: f32,     // subsampling stride (cast to u32)
  lens_enabled: f32,  // > 0.5: multiply by the vignetting gain
  centre_x: f32,      // optical centre, in texture pixels
  centre_y: f32,
  half_diag: f32,     // pixels at rho = 1
  domain: f32,        // rho covered by the lens table
  pad0: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
};

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 1024>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> lens_table: array<vec4f>;

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

// The vignetting gain at a source texel: the table's .w at its radius, linearly interpolated and held at the ends.
fn lens_gain(texel: vec2u) -> f32 {
    let n = arrayLength(&lens_table);
    let last = f32(n - 1u);
    let rho = length(vec2f(texel) + 0.5 - vec2f(params.centre_x, params.centre_y)) / params.half_diag;
    let x = clamp(rho / params.domain * last, 0.0, last);
    let i = min(u32(floor(x)), n - 2u);
    let a = lens_table[i].w;
    return a + (lens_table[i + 1u].w - a) * (x - f32(i));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(t_input);
    let s = u32(params.stride);
    let coord = gid.xy * s;
    if (coord.x >= dims.x || coord.y >= dims.y) { return; }

    var c = textureLoad(t_input, vec2i(coord), 0).rgb;
    if (params.lens_enabled > 0.5) { c *= lens_gain(coord); }

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
