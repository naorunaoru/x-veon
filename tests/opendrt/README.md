# OpenDRT conformance checks

The renderer targets the **OpenDRT v1.0 transform**, using the bundled
`opendrt_art.ctl` as its complete numerical reference. It does not implement
OpenDRT v1.1's redesigned color modules.

The four **OpenDRT 1.0** look options contain the original Default, Colorful,
Umbra and Base parameters from Jed Smith's
[v1.0.0 StickShift release](https://github.com/jedypod/open-display-transform/releases/tag/v1.0.0).
Existing look IDs and their parameter values are retained for saved photos;
the overlapping names are labelled **X-Veon** to distinguish those adaptations.
The corrections to hue handling and export processing apply to all looks.

## Run

Install the normal web dependencies, then use a Python environment with `numpy`
and `wgpu` installed. Node, a C++ compiler and a working WebGPU adapter are required.

```sh
npm --prefix web run test:opendrt
npm --prefix web test
```

The GPU test compiles the **actual production WGSL module**, adds a compute entry
point that calls `opendrt()`, and packs uniforms using the production TypeScript
code. It runs 4,010 deterministic colors across all presets in Rec.709, P3 and
P3-limited Rec.2020 at 100 and 1,000 nits, explicit regression cases and 1,000
seeded parameter combinations. It rejects nonfinite output and black middle grey.

Reference comparisons compile the complete bundled CTL transform as C++ with its
math unchanged, including hue shifts, hue contrast and creative white. Three CTL
helper names are changed only to avoid standard-library collisions. The source
input conversion uses the same project sRGB→P3 matrix before the CTL's P3 input
path. The absolute per-channel tolerance is `1e-5`, in peak-normalized display-linear
RGB. Fractional warmth is a project extension; its monotonic behavior is tested
separately from the reference's D65/D50 endpoints.

Recorded on Apple M4 Pro / Metal: 1,091 configurations, 85 complete reference
comparisons, maximum absolute channel error `3.3974648e-6`.

## Intentional integration differences

- Input is linear sRGB from the camera pipeline. The project retains its existing
  sRGB→P3 matrix for compatibility with that pipeline.
- Preview applies the sRGB transfer function. HDR preview uses extended P3 and
  a display scale of 1; exports are normalized to their configured peak. The
  numerical reference comparisons happen before the preview transfer function.
- Warmth is a continuous D65→D50 matrix blend; upstream uses a whitepoint enum.
  Range zero still applies highlight-weighted warmth, and range one applies it
  throughout the image, as in upstream.
- Invalid saved render strengths and zero range divisors are normalized at render
  time. Low-contrast width zero uses its identity limit. Achromatic hue is guarded
  because WGSL does not define `atan2(0, 0)`.

The old `opendrt_reference.py` and JSON vectors are historical tests for an earlier
Rust implementation. That Python translation omits hue shifts, hue contrast and
creative white; it must not be used to establish full shader conformance.
