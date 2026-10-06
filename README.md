# What-veon?

X-veon: neural network demosaicing for Bayer and X-Trans sensors. 

This project consists of two parts: first one is the neural net itself with a bunch of scripts for dataset building and training, the other is a web application with a full RAW development pipeline.

## Neural network

The demosaicing model is a U-Net (encoder-decoder with skip connections, `model.py`) with a 5-channel input: the raw CFA mosaic value, 3 binary masks marking which colour filter covers each pixel, and a clip-proximity channel (0 below half of the clip level, ramping to 1 at clipping). It outputs a full-colour 3-channel image in camera RGB, without white balance.

The encoder has 4 downsampling stages (strided convolutions; channel widths `base_width × 1, 2, 4, 8, 16`). Each stage is two convolutions with GroupNorm and ReLU; the decoder upsamples with 1×1 convolutions and PixelShuffle and concatenates the matching encoder stage. For X-Trans the first convolution is 7×7 and the input also carries sin/cos encodings of the 6×6 CFA phase.

A key design choice is the residual CFA skip: each photosite's value is placed in its own colour channel as a baseline (`cfa × masks`), and the network learns the missing colours on top of it. This keeps the model largely exposure-agnostic.

The same architecture serves both 6×6 X-Trans and 2×2 Bayer patterns, with a separate model per sensor type. The models shipped in `shared/public/checkpoints/` were exported before the current architecture (max-pool/transposed-convolution, no normalisation), so the current `model.py` cannot load them; retrain to reproduce them.

## Dataset

The network is trained on synthetic input/target pairs generated from real RAW photos. The build process works as follows:

1. **Ground truth generation** (`build_dataset.py`): RAW files (RAF, ARW, CR2, etc.) are demosaiced with traditional algorithms — DHT for X-Trans, AHD for Bayer — in linear sensor space with no white balance or colour correction, normalised to the sensor's range, and downscaled 2× by area averaging to produce clean reference images stored as float32 `.npy` files.

2. **Synthetic re-mosaicing**: during training, patches are cropped on the CFA period from the ground truth and re-mosaiced through the sensor's pattern to form the network's input, so the model learns from a clean demosaic "re-captured" through the CFA.

3. **Augmentations**: random flips and 90° rotations, Poisson-Gaussian noise, optional OLPF (anti-aliasing filter) blur, synthetic bright light sources pushing into clipping, random downscaling, and — for models trained on white-balanced input (`--apply-wb`) — white-balance perturbation in log space.

4. **Torture patterns**: a fraction of synthetic gradient and edge patterns can be mixed into the training set (`--torture-fraction`) to improve worst-case inputs like fine diagonal lines and colour fringes near Nyquist.

## Web application

A small, fully offline (as in all processing is done in the browser) web application was built along the model. It uses ONNX WebGPU runtime for inference, so a decent GPU is required. Processing times on an M1 Macbook Pro are in the tens of seconds at worst.

### Live demo:

Stable: https://naorunaoru.github.io/x-veon

Beta (new UI and pipeline, separate library): https://naorunaoru.github.io/x-veon/beta/

What it can do:
- open RAW files from different cameras, tested mainly on Fujifilm RAFs and Sony ARWs
- perform neural net or traditional numeric demosaicing for comparison
- limited color grading creative controls
- preview and save HDR photos

Supported output formats: 
- UHD JPEG: 3-channel gain map, works best
- AVIF is super slow and has incorrect gamma, which can be solved by moving from HLG to PQ
- uncompressed 16-bit TIFF is there too

What it can't do yet:
- export as DNG
- passthrough full EXIF metadata
- do batch operations

### Development

The RAW decoder is a git submodule (`shared/crates/vendor/rawloader`, a pinned fork of rawloader). Clone with `git clone --recurse-submodules`, or run `git submodule update --init` in an existing checkout. Then, from the repository root:

```
npm run setup          # rust wasm target, npm install, wasm builds
npm run build:lensfun  # lens-correction data (gitignored, needed once)
npm run dev
```

`npm run dev` serves without cross-origin isolation, like GitHub Pages.

`npm run build` and `npm run preview` need `XV_CHANNEL` (`stable`, `beta` or `dev`); see `RELEASING.md`. CI builds the site with `scripts/build-web.sh <channel>`, which you can run too.

Layout: `shared/` holds everything the app runs: `shared/src` (TypeScript), `shared/crates` (Rust, built to WASM) and `shared/public` (models, lens data, samples). `web/` is the web host: its entry file, `index.html` and the Vite config for GitHub Pages. The npm and Cargo workspaces are declared at the root.

Source layout (`shared/src`): `lib` (shared types and the method/format catalogue, no dependencies) ← `gpu` (the shared WebGPU device) ← `pipeline` (RAW → processed image, framework-free) and `renderer` (display, histogram, readback) ← `app` (the store as pure slices, persistence/library/processing/export/bootstrap services, hooks) ← `components` (React UI). Imports only go left; `shared/src/test/layers.test.ts` fails the suite on a violation and names the file and line.

#### Moving an existing checkout

A checkout from before the move to `shared/` + `web/` (September 2026) needs a one-time cleanup after it pulls the new layout. Commit or set aside work in progress first: edits to files under `web/src` follow them into `shared/src`, and a file you added there comes back as a file-location conflict with a suggested `shared/src/…` path. Then, from the repository root:

<!-- existing-checkout:begin -->
```bash
git submodule update --init
mkdir -p shared/public
for d in samples lensfun; do
  if [ -d "web/public/$d" ] && [ ! -e "shared/public/$d" ]; then mv "web/public/$d" shared/public/; fi
done
if [ -d web/.lensfun-db ] && [ ! -e shared/.lensfun-db ]; then mv web/.lensfun-db shared/; fi
rm -rf web/wasm web/node_modules web/dist web/.lensfun-db
find web/public -depth -type d -empty -delete 2>/dev/null || true
npm run setup
```
<!-- existing-checkout:end -->

This keeps your samples and lens data and removes the old layout's build output and its copy of the decoder submodule, which now lives at `shared/crates/vendor/rawloader`. Untracked files of your own under `web/src` stay where they were; move them into `shared/src`. Checking out a commit from before the move again, such as `main` until its next promotion, leaves `shared/`, `node_modules/` and `target/` untracked: delete them there, or switch back.

## Desktop beta

The native encoder needs Rust stable. On macOS, install the Xcode command-line tools. On Windows, install the MSVC build tools and NASM 2.15 or later, with NASM on `PATH`.

```bash
npm run build:native --workspace desktop
npm run test:native --workspace desktop
```

Run `npm run build:wasm` before the native tests when running the parity test, which compares the native and WASM encoders.

Set up the repository as described above, then run the desktop app from the repository root:

```bash
npm run build --workspace desktop
npm run start --workspace desktop
```

Build a local installer with `npm run dist --workspace desktop -- --mac` on macOS or `npm run dist --workspace desktop -- --win` on Windows. The `dist` command builds the native addon before bundling the app; Windows requires NASM on `PATH`. The macOS build produces an arm64 DMG; the Windows build produces an x64 NSIS installer. Run `node desktop/scripts/check-dist.mjs` after the macOS build to check archive contents, the size budget, and the unpacked native addon. These local builds are unsigned for distribution. On macOS, if the first launch is blocked, use System Settings → Privacy & Security → **Open Anyway**. On Windows, if SmartScreen appears, select **More info** → **Run anyway**.

Run the packaged-app smoke test after building with `npm run dist --workspace desktop -- --mac --dir`:

```bash
XV_SMOKE_APP="$PWD/desktop/dist/mac-arm64/X-veon Beta.app/Contents/MacOS/X-veon Beta" \
XV_SMOKE_SAMPLES="$PWD/shared/public/samples" npm run test:smoke --workspace desktop
```

`XV_SMOKE_APP` is the packaged executable (on Windows, `desktop/dist/win-unpacked/X-veon Beta.exe`). `XV_SMOKE_SAMPLES` contains `DSCF3332.RAF` and `sony_a6400_21.arw`; the test edits temporary copies. No browser download is needed. For a tagged build, also set `XV_SMOKE_TAG` to its tag. Every run answers main's update check from a local server, so the test never asks GitHub. Set `XV_SMOKE_REPORT` to choose the JSON report path; failed runs retain a trace in `desktop/test-results/`.

Desktop edits are saved beside each RAW in a `.xmp` sidecar. Other apps may drop the `xveon:` properties when they rewrite that sidecar, so keep a copy if you edit the same photo in another app.

## License

This project uses a multi-license structure:

| Component | License | SPDX Identifier |
|---|---|---|
| Neural network code (model, training, losses, dataset) | MIT | `MIT` |
| Trained model weights and ONNX checkpoints (`shared/public/checkpoints/`) | Creative Commons Attribution 4.0 | `CC-BY-4.0` |
| Processing pipeline, web app, and everything else | GNU GPL v3 or later | `GPL-3.0-or-later` |

See [LICENSE](LICENSE) for details and [LICENSES/](LICENSES/) for full license texts.

## Acknowledgments

RAW decoding uses a fork of [rawloader](https://github.com/pedrocr/rawloader) (LGPL-2.1), included as the `shared/crates/vendor/rawloader` submodule.

Parts of the code were adapted from various open-source projects:
- darktable (segmentation-based highlight reconstruction, reference image pipeline)
- Jed Smith's OpenDRT and ART CTL by agriggio (tone mapping)


## Photo edits and browser storage

Settings has separate controls for the selected photo and for defaults. Each photo records its demosaic method and neural model checkpoint; changing defaults does not change an explicit photo edit. If a saved checkpoint is unavailable, the app shows a note and renders with the available model of the same size, or the default size when necessary. Its recorded checkpoint changes when you edit that photo again.

A photo can be **saved**, **session** (editable, but its last save failed), or **view only**. Session edits remain in memory and retry on the next change or when the window regains focus. The UI shows the reason for session or view-only state.

**Clear library** in Settings removes this channel's photos, edits and settings, then reloads. Stable also removes storage left by the old app. Other channels and other sites on the same origin are preserved. If clearing is blocked, close other tabs using the library and retry. After a failed clear, imports and persistence stay paused until clearing succeeds or the page is reloaded, so pending writes cannot recreate deleted data.

The app no longer requests persistent browser storage. A browser's existing persistence grant remains until its own storage controls remove it.

### Host boundary

`shared/src/host/` defines the host contracts. Shared startup is `startApp(root, host)`; browser library storage, encoder workers, downloads and HDR detection live in `web/src/host/`. Shared app settings use the host-supplied IndexedDB name. `npm test` runs both workspace suites; `npm test --workspace web` runs just the browser adapters. The layers test enforces the workspace boundary and prevents host detection in shared code.

### Desktop filesystem validation

`npm test --workspace desktop` runs the native watcher separately after the other desktop tests, keeping its 3-second deadline. A central file-symlink probe skips only file-link cases on Windows `EPERM` (missing privilege); record the explicit reason and skipped case count. Directory swaps use junctions. Unexpected capability errors fail. No Developer Mode change is needed.

For Windows write-denial acceptance, apply an actual ACL denial to a copied fixture folder, then restore its original ACL. A read-only attribute is not equivalent. Viewing creates no probe sidecars; the real save/reset result determines session state. Record the OS error and retained edit. macOS tests do not prove Windows ACL behavior.

Use `--user-data-dir=/absolute/isolated-profile` for a separate desktop test/golden profile, including its settings, cache and single-instance lock.

The export encoder core is `shared/crates/encode`; its browser WASM wrapper builds in `web/crates/encode-wasm` and generates its package in that crate’s `pkg/` directory. `npm run build:wasm` builds the shared decoder and demosaic crates and the web encoder wrapper.
