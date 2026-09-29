# M2 Electron spike

This branch starts at `a738f131c1d782b128138af4448f5f5cf3459325`. It is a fixture-only
experiment, with no folder library, sidecars, native addon, installer or desktop export.
Edits stay in memory. Keep the spike branch for the rest of M2 after findings and the
next plan are reviewed; do not merge this spike separately. Remove diagnostic routes,
fixture assets and report writing before any distributable build.

## Windows transfer and setup

Use **Git Bash**, Node **24**, Rust with `wasm32-unknown-unknown`, and Chrome. Use x64
Node on Windows x64. The package's `TRANSFER.md` supplies the exact spike commit and
checksums. The parent bundle contains history and models, but not submodule objects,
RAW fixtures or reports. Network access is needed for npm and the HTTPS rawloader
submodule. Use a fresh clone to avoid existing CRLF conversions:

```bash
handoff_dir=/c/path/to/m2-handoff
cd "$handoff_dir"
sha256sum -c SHA256SUMS
# -c before clone also covers submodule clones; clone -c persists it locally.
git -c core.autocrlf=false clone -c core.autocrlf=false --recurse-submodules \
  -b codex/desktop-m2-spike "$handoff_dir/m2-spike.bundle" x-veon-m2
cd x-veon-m2
git rev-parse HEAD                 # must match TRANSFER.md
git submodule status              # pinned rawloader, no leading + or -
git ls-files --eol scripts/build-web.sh  # must show w/lf
mkdir -p shared/public/samples
cp "$handoff_dir/fixtures/DSCF3332.RAF" shared/public/samples/
cp "$handoff_dir/fixtures/sony_a6400_21.arw" shared/public/samples/
cp "$handoff_dir/fixtures/manifest.json" shared/public/samples/
(cd shared/public/samples && sha256sum -c "$handoff_dir/FIXTURE-SHA256SUMS")
printf '\n/.spike-evidence/\n' >> "$(git rev-parse --git-path info/exclude)"
git check-ignore .spike-evidence/probe.json
node --version
rustc --version
rustup target add wasm32-unknown-unknown
bash scripts/build-web.sh beta
```

The web recipe installs/tests all three workspaces and skips the Electron binary
only for that invocation. Install the pinned Electron binary, then build both local
spike runtimes. These commands also work on macOS:

```bash
unset ELECTRON_SKIP_BINARY_DOWNLOAD
node node_modules/electron/install.js
XV_CHANNEL=dev XV_GOLDEN=1 npm run build --workspace web
npm run build --workspace desktop
node desktop/scripts/serve-web.mjs
```

Keep the server terminal open. It binds only `127.0.0.1:39217`, prints/saves its PID,
and writes browser JSON to `tmp/m2-spike/browser/<report>.json`. It serves the built
web directory, including the copied fixtures. All `tmp/` evidence is git-ignored.
Use unique report/run names matching `[a-z0-9-]+`; repeated names overwrite reports.

## Golden and HDR

In Chrome, open the following URLs, waiting for the title/report to finish:

- `http://127.0.0.1:39217/?golden=full&report=chrome-full-1`
- `http://127.0.0.1:39217/?golden=full&report=chrome-full-2`

Each full run requires ten processing cases and six exports. Save the two reports.
For a new adapter, expect `RECORDED`, not `PASS`. Compare every hash, follow
`shared/src/test/golden/README.md`, and obtain user sign-off before adding the Windows
baseline. Do not replace the Apple baseline or widen tolerances. Rebuild both hosts
after adding the signed-off baseline; rerun Chrome full and Electron rendering.

In another Git Bash terminal at the repository root:

```bash
npm run start --workspace desktop -- --spike=golden --spike-run=windows-golden
```

Electron runs ten rendering cases without exports. Reports are
`tmp/m2-spike/runtime/windows-golden-{capabilities,golden}.json`. Errors produce
`<run>-error.json`. Each launch records `<run>-pid.json`; close the window before
starting another run. Only stop task-owned PIDs. A golden report needs `PASS` with
all ten cases; an error report or missing cases fail. Compare all three image hashes
against that machine's Chrome full report as well as the selected baseline.

The capability report must show a secure `app://bundle` page, no page Node access,
`crossOriginIsolated: false`, sandbox/context isolation enabled, node integration
disabled, the actual pipeline adapter and WebGPU backend. Record Chrome's running
version and `chrome://gpu` physical GPU/driver alongside Electron's full GPU info.
On a dual-GPU machine both must use the same physical GPU before comparisons count.
Electron is pinned to 44.4.5 (Chromium 152); record any Chrome major-version gap.

For physical HDR comparison, use the same display, HDR enabled, power state, brightness,
fixture and grading settings. Run `--spike=view --spike-sample=xtrans` in Electron and
open `http://127.0.0.1:39217/?spike=view&sample=xtrans` in Chrome. Repeat with `bayer`.
Inspect the same bright areas in each application; record the display, observed match
and user acceptance. Golden reports include accepted canvas configuration. Configuration
and hashes alone do not prove HDR presentation. If no HDR display is available, mark
this gate pending. The spike uses the shared media-query fallback, not native headroom.

## Timing

Keep the machine plugged in with the same power mode, stable temperature and no competing
GPU workload. Record those conditions, OS/build, GPU/driver, display/HDR state, running
Chrome/Electron versions, commit, and any launch flags. Reports include the active
adapter, neural model SHA, dimensions and backend. Do not infer the running Chrome
version from an application update waiting for restart.

Alternate runtime order: Chrome X-Trans, Electron X-Trans, Electron Bayer, Chrome Bayer.
Close Electron between invocations. Each batch uses a fresh page/process, records first
processing separately, discards one warm-up, then measures five runs. “First” means first
processing in that launch; disk and driver caches may already be warm. Only one runtime
should be processing at a time.

```text
Chrome: http://127.0.0.1:39217/?spike=timing&sample=xtrans&report=chrome-xtrans
Electron: npm run start --workspace desktop -- --spike=timing --spike-sample=xtrans --spike-run=electron-xtrans
Electron: npm run start --workspace desktop -- --spike=timing --spike-sample=bayer --spike-run=electron-bayer
Chrome: http://127.0.0.1:39217/?spike=timing&sample=bayer&report=chrome-bayer
```

`inferenceMs` measures neural demosaic strategy execution through GPU completion,
excluding model activation and postprocessing. `processingMs` covers a RAW byte copy,
decode/preparation, activation, demosaic and postprocessing through GPU completion.
File fetch is excluded; `initializationWaitMs` is the wait for app readiness, not a
complete startup benchmark. The legacy app's combined timer is retained separately
as `demosaicPostprocessMs`.

For each fixture report all five measured values for both metrics, medians and the
Electron/Chrome median ratio. The inference gate is Electron median <= Chrome median
+ (Chrome maximum - Chrome minimum). Show that spread in ms and percent. If it fails,
repeat both batches once and preserve both sets. Drift/outliers can make the result
inconclusive; never use timing noise as an image-hash tolerance.

## Utility-process transport

```bash
npm run start --workspace desktop -- --spike=transport --spike-run=windows-transport
```

The report is `tmp/m2-spike/runtime/windows-transport-transport.json`. Main only brokers
the port. The renderer sends deterministic bytes directly to the utility worker: exactly
400,000,000 bytes, chunks <=64,000,000, one outstanding chunk, all bytes verified.
Allocation/fill precedes timing; chunk copies and acknowledgements are timed. There is
one warm-up plus five measured transfers. Receipt and verified times are separate.
Worker RSS/external memory and app process metrics are observations, not peak-memory
measurements. Garbage collection may retain allocations after references are released.

The harness first interrupts an active request by killing the worker; its port must
close, and a fresh worker must complete subsequent transfers. Silent requests time out.
An eight-byte probe records transfer-list support and actual sender detachment; a second
probe checks structured cloning. On the Mac's Electron 44.4.5, a transferred ArrayBuffer
arrived as null, while cloning succeeded. The bulk test therefore used cloned chunks.
This keeps the intended process/port route but measures copies; do not call it zero-copy.
Report either behavior on Windows. No throughput threshold is prescribed.

## Return and review gates

Create `codex/desktop-m2-windows` from the supplied commit. Stage explicit paths for
reviewed Windows fixes and the signed-off baseline. Keep evidence out of Git. Then:

```bash
git bundle create "$handoff_dir/m2-windows.bundle" codex/desktop-m2-windows ^<original-spike-sha>
git bundle verify "$handoff_dir/m2-windows.bundle"
git rev-parse HEAD
sha256sum "$handoff_dir/m2-windows.bundle"
```

Return that bundle, original/returned SHAs, raw JSON/logs, checksums and a short report
covering every gate. If there are no commits, return evidence and baseline/sign-off
status without making an empty bundle. Mac verifies/fetches into a review branch before
integration. Changes to shared code require fresh comparable reports from both machines
at the same final commit.

The spike remains pending until both machines and the physical HDR checks are complete.
The changed web recipe also needs a real Linux dry run before develop integration.
Push and workflow dispatch each need separate user approval, after a concrete candidate
commit and local checks are ready. The approved dispatch shape is:

```bash
gh workflow run deploy.yml --ref develop -f dry_run=true -f beta_ref=<candidate-sha>
```

No push, dispatch, merge, tag or release is authorized by these instructions alone.
