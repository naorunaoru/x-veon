# Golden baseline

`baseline.json` holds SHA-256 hashes of the renderer's scene texture, the app's graded readbacks
and its encoded exports for two sample RAWs, produced by the `?golden` route in
`src/dev/golden.ts`. The route is compiled in on the dev server and in builds made with
`XV_GOLDEN=1`; normal builds do not contain it.

## Provenance of the current baseline

Re-recorded 2026-09-27 on Apple `metal-3` in Chrome 153, at `develop` `1c24dc6`, before the repository moved into `shared/` + `web/` (desktop M0). Two full runs agreed on every scene, graded-readback and export hash. The whole file was regenerated, so it supersedes the stale graded and export hashes described in the sections below. Against the 2026-09-04 baseline, the scene hashes changed for both neural-net runs and for every X-Trans run, which fits `88dd3f7` (no black first row/column in neural-net output) and `7780116` (black and white levels by CFA colour); the four Bayer traditional runs are unchanged.

## Earlier baseline (2026-09-04)

Recorded 2026-09-04 on Apple `metal-3` at commit `5b064b4`, which is the pre-consolidation tree
(`develop` at `3ecf71e`) plus the golden harness (`6e5b215`) plus the deterministic chroma
reduction (`2593aa0` on the feature branch is the cherry-pick of that fix). `5b064b4` is on no
branch; rebuild the baseline tree with:

```bash
git checkout 6e5b215 && git cherry-pick 2593aa0
```

2026-09-04: `scene` hashes recorded from Plan A commit `526aab3` plus readback instrumentation,
before the Plan B production refactor. Existing display/displayDark hashes matched and were left
untouched. Scene equality covers Plan B; it does not retroactively prove scene equality across
Plan A.

Hashes are adapter-specific. A mismatch on another GPU is expected; a mismatch on the recording
adapter is a defect to explain, never a tolerance to widen.

## OpenDRT correction, 2026-09-05

The OpenDRT conformance fixes intentionally change graded readbacks and encoded
exports (hue remainder semantics, achromatic handling, and P3-limited Rec.2020
processing). The hashes below remain the historical pre-fix baseline; they have
not been regenerated because `public/samples/` is absent from this checkout.
Scene texture hashes should remain unchanged. Re-record the graded/export hashes
with the two required RAW samples using the procedure below before treating this
browser suite as a current rendering baseline. Do not widen comparisons to hide
the intended changes. The standalone `tests/opendrt/check_shader.py` checks the
corrected shader against the complete CTL reference without those RAW samples.

## Export precision, 2026-09-25

Exports now always render to `rgba32float`. They previously used `rgba16float` whenever the
device lacked `float32-blendable`, which ONNX Runtime's shared device never requests, so the
recorded export hashes (already stale above) will not match either. Re-record together.

## Running

```bash
npm run build:wasm
XV_GOLDEN=1 XV_CHANNEL=dev npm run build --workspace web
XV_CHANNEL=dev npm run preview --workspace web -- --port 4190
```

Then open `http://localhost:4190/?golden` (quick: neural S plus one traditional method per
sample, two runs, no exports) or `http://localhost:4190/?golden=full` (every applicable method
once, plus Ultra HDR JPEG, AVIF and TIFF exports of the neural S result). The page title ends
in the overall status; the report is printed at the bottom of the page and stored in
`window.__golden` as `{ status, results, report, expected }`.

Samples live in the gitignored `shared/public/samples/` with a `manifest.json` that must match the
contract in `golden.ts` exactly:

```json
{ "samples": [
  { "file": "DSCF3332.RAF", "cfa": "xtrans", "traditional": "dht" },
  { "file": "sony_a6400_21.arw", "cfa": "bayer", "traditional": "ahd" }
] }
```

## Recording a new baseline

Only when the pipeline's output is meant to change. Delete `baseline.json`, run the full route
twice, and accept only if every hash agrees between the two runs: status `RECORDED` both times,
every entry `stable`, and identical hash fields, `scene`, `display` and `displayDark` for each entry
and `bytes` and `sha256` for each export. Entries also carry `elapsedMs` and `runs`, which vary
between runs and don't belong in the baseline. Write those hash fields into `baseline.json` together
with `adapter`, `commit` and `recordedAt` from the report, and note the reason here.
