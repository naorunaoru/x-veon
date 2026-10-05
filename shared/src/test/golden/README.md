# Golden baseline

`baselines/*.json` hold SHA-256 hashes of the renderer's scene texture, the app's graded readbacks
and its encoded exports for two sample RAWs, produced by the `?golden` route in
`src/dev/golden.ts`. The route is compiled in on the dev server and in builds made with
`XV_GOLDEN=1`; normal builds do not contain it.

## Provenance of the current baseline

### Windows NVIDIA Blackwell (2026-10-01)

Recorded on an RTX 5080, driver 32.0.16.1714, Windows 11 build 26300, Chrome
154.0.8037.58 at `6bae195`. Two full HDR-enabled Chrome runs agreed on all 30
scene/graded/dark hashes and all six export hashes and byte lengths. The user
explicitly approved this baseline on October 1. Evidence is preserved in the
Windows spike reports `windows-chrome-hdr-full-1.json` and
`windows-chrome-hdr-full-2.json`. The Apple baseline is unchanged.

### Apple Metal

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

## Host routing

Golden exports use the shared export service and `host.exporter`, including the real web WASM encoder. The web entry disables download delivery for a compiled-in golden route; the harness hashes the returned bytes without saving files into Downloads. The baseline and its adapter-specific comparison rules are unchanged by M1.

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
contract in `shared/src/dev/golden-contract.ts` exactly:

```json
{ "samples": [
  { "file": "DSCF3332.RAF", "cfa": "xtrans", "traditional": "dht" },
  { "file": "sony_a6400_21.arw", "cfa": "bayer", "traditional": "ahd" }
] }
```

## Recording a new baseline

For a new GPU adapter, run Chrome's full route twice without adding a baseline first. Both
reports must be `RECORDED`, every entry stable, and all scene/display/displayDark hashes
and export bytes/sha256 identical across the two runs. Obtain the user's sign-off before
committing `baselines/<adapter-label>.json`. Keep the other adapters' files. A deliberate
output change on an existing adapter needs the same two-run comparison and sign-off;
move only that adapter's existing file outside the source tree while recording.

Copy `adapter`, `commit`, `recordedAt`, and the hash fields into the baseline. Exclude
`elapsedMs` and `runs`, which vary. Record the reason and provenance here.

## Adapter selection and desktop spike

The loader selects exactly one baseline by the pipeline device's `adapter.vendor` and
`adapter.architecture`; filenames are labels only. An unknown adapter is an error. A
missing baseline gives `NEW`/`RECORDED`, never `PASS`. Multiple matching baselines are an
error. There is no fallback to the Mac baseline. If two physical devices collide on this
identity pair, extend the report and selection identity before adding the second baseline.

The original Apple baseline moved unchanged to `baselines/apple-metal-3.json` in M2.
Its provenance above still applies. Windows recording must use the same physical GPU as
Electron, with driver and runtime versions saved alongside the reports.

`?golden=render` runs all ten processing cases once and checks exact scene, graded and
dark-graded hashes. It does not exercise exports. Electron defaults to `?golden=full`, with all six exports
through the native encoder. Use `--golden-mode=render` for the old render-only route.
`--golden-mode=bench` is reserved for the Task 10 benchmark workflow. `?golden=full` continues to
require all ten cases and all six exports. Missing expected cases fail in either mode.
See `README.md` and `desktop/scripts/` for built Electron commands. Acceptance and cross-machine evidence live in the M2 plan and handoff.

Run the native Electron gate on macOS with an isolated profile:

```bash
npm run build:native --workspace desktop
XV_GOLDEN=1 npm run build --workspace desktop
npx electron desktop --golden-report="$PWD/.m3-evidence/task9-electron-golden.json" --user-data-dir="$(mktemp -d)"
```

Main creates `<report>.exports` for fixed native destinations. The harness compares the
worker's SHA-256 and byte receipts against the unchanged adapter baseline.
