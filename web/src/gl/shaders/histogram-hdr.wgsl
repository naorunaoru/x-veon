// Compute shader: bins a subsampled rgba32float scene-linear texture
// into 4 × 256 histogram bins in log2 EV space [-8, +8].
// Dispatched over (width/STRIDE × height/STRIDE), samples every 4th pixel.
// Layout: bins[0..255] = R, bins[256..511] = G, bins[512..767] = B, bins[768..1023] = L (Rec.709)

struct Params {
  exposure: f32,
  wb_temp: f32,
  wb_tint: f32,
};

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 1024>;
@group(0) @binding(2) var<uniform> params: Params;

const EV_MIN: f32 = -8.0;
const EV_RANGE: f32 = 16.0;
const STRIDE: u32 = 4u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(t_input);
    let coord = gid.xy * STRIDE;
    if (coord.x >= dims.x || coord.y >= dims.y) { return; }

    var c = textureLoad(t_input, vec2i(coord), 0).rgb;

    // Exposure + white balance (matching opendrt.wgsl preprocessing)
    c *= exp2(params.exposure);
    c.x *= exp2(params.wb_temp);
    c.y *= exp2(-params.wb_tint);
    c.z *= exp2(-params.wb_temp);
    // Tint luminance compensation
    c /= 0.2126 * exp2(params.wb_temp) + 0.7152 * exp2(-params.wb_tint) + 0.0722 * exp2(-params.wb_temp);

    // Per-channel log2 EV, mapped to [0, 255] over [-8, +8] EV
    let r_ev = log2(max(c.x, 1e-10));
    let g_ev = log2(max(c.y, 1e-10));
    let b_ev = log2(max(c.z, 1e-10));
    let l_ev = log2(max(0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z, 1e-10));

    let ri = u32(clamp((r_ev - EV_MIN) / EV_RANGE * 256.0, 0.0, 255.0));
    let gi = u32(clamp((g_ev - EV_MIN) / EV_RANGE * 256.0, 0.0, 255.0));
    let bi = u32(clamp((b_ev - EV_MIN) / EV_RANGE * 256.0, 0.0, 255.0));
    let li = u32(clamp((l_ev - EV_MIN) / EV_RANGE * 256.0, 0.0, 255.0));

    atomicAdd(&bins[ri], 1u);
    atomicAdd(&bins[256u + gi], 1u);
    atomicAdd(&bins[512u + bi], 1u);
    atomicAdd(&bins[768u + li], 1u);
}
