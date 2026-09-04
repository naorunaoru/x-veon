// Single-workgroup compute shader that scans histogram bins to find:
//   - occupied bin range (lo, hi) with padding
//   - max bin count (excluding extreme bins 0 and 255) as log(1 + max)
// Output feeds the histogram visualization shader.

struct Result {
  bin_lo: f32,
  bin_hi: f32,
  max_log: f32,
  pad: f32,
};

@group(0) @binding(0) var<storage, read> bins: array<u32, 1024>;
@group(0) @binding(1) var<storage, read_write> result: Result;
@group(0) @binding(2) var<uniform> force_zero_lo: u32;  // 1 for rgb/luma, 0 for ev

var<workgroup> wg_lo: atomic<u32>;
var<workgroup> wg_hi: atomic<u32>;
var<workgroup> wg_max: atomic<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  let tid = lid.x;

  // Initialize workgroup atomics
  if (tid == 0u) {
    atomicStore(&wg_lo, 256u);
    atomicStore(&wg_hi, 0u);
    atomicStore(&wg_max, 0u);
  }
  workgroupBarrier();

  // Each thread scans 4 bins (one per channel at the same bin index)
  if (tid < 256u) {
    var any_nonzero = false;
    var local_max = 0u;

    for (var ch = 0u; ch < 4u; ch++) {
      let val = bins[ch * 256u + tid];
      if (val > 0u) {
        any_nonzero = true;
      }
      // Exclude extreme bins 0 and 255 from max calculation
      if (tid > 0u && tid < 255u && val > local_max) {
        local_max = val;
      }
    }

    if (any_nonzero) {
      atomicMin(&wg_lo, tid);
      atomicMax(&wg_hi, tid);
    }
    atomicMax(&wg_max, local_max);
  }

  workgroupBarrier();

  if (tid == 0u) {
    var lo = atomicLoad(&wg_lo);
    var hi = atomicLoad(&wg_hi);
    let max_count = atomicLoad(&wg_max);

    if (hi < lo) {
      lo = 0u;
      hi = 255u;
    } else {
      let span = max(hi - lo, 1u);
      let pad = max(2u, u32(round(f32(span) * 0.05)));
      if (lo > pad && force_zero_lo == 0u) {
        lo -= pad;
      } else if (force_zero_lo != 0u) {
        lo = 0u;
      }
      hi = min(255u, hi + pad);
    }

    result.bin_lo = f32(lo);
    result.bin_hi = f32(hi);
    result.max_log = log(1.0 + f32(max_count));
    result.pad = 0.0;
  }
}
