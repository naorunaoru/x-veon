# What-veon?

X-veon: neural network demosaicing for Bayer and X-Trans sensors. 

This project consists of two parts: first one is the neural net itself with a bunch of scripts for dataset building and training, the other is a web application with a full RAW development pipeline.

## Neural network

The demosaicing model is a small U-Net (`model.py`) with a 5-channel input: the raw CFA mosaic value, 3 binary masks marking which colour filter covers each pixel, and a clip-proximity channel (0 below half of the clip level, ramping to 1 at clipping). It outputs a full-colour 3-channel image in camera RGB, without white balance.

The model divides the mosaic by its mean over the tile, so it sees every tile at the same brightness, and multiplies its output back at the end. Exposure therefore does not change the result, down to a tile mean of 1e-4 of the sensor's range.

The input is then packed by space-to-depth, 3×3 for X-Trans and 2×2 for Bayer, so each channel holds one photosite position and every convolution sees a fixed layout. The S model works at two resolutions (1/3 and 1/6 for X-Trans, 1/2 and 1/4 for Bayer). A block is two 3×3 convolutions with ReLU; there are no normalisation layers. A 1×1 convolution and depth-to-space turn the result into a full-resolution correction.

That correction is added to a baseline in which each photosite's value sits in its own colour channel (`cfa × masks`), so the network only supplies the missing colours.

The same code serves both 6×6 X-Trans and 2×2 Bayer patterns, with a separate model per sensor type. `CHECKPOINT_POLICY.md` describes checkpoint versions; the current family is `v7`. In `shared/public/checkpoints/models.json`, an entry exported from the current code carries a `checkpoint_version`; an entry without one is an older export that this `model.py` cannot load.

## Training

`train.py` trains with L1 between prediction and target after both are divided by the target patch's mean and passed through a power curve (γ = 1/2.2, with a small offset that keeps the slope at black finite). The S configuration adds one more term on the same encoded values: L1 between their magnitude spectra (weight 0.5), which does not care where fine texture sits but charges for texture that is missing, so faint detail is not averaged away. The encoding also gives the PSNR that is reported and that picks `best.pt`, so its values are not comparable with PSNR figures from earlier checkpoints.

`configs/s_xtrans/config.json` and `configs/s_bayer/config.json` hold the S training configuration:

```
python train.py --from-checkpoint configs/s_xtrans --no-resume \
    --data-dir <dataset>:1500 <dataset>:1500 \
    --output-dir checkpoints/xtrans/v7.1.0 --cache-patches --cache-gb 40 --workers 4
python export_onnx.py --version v7.1.0 --verify
```

`tools/eval_truth.py` scores exported models against real RGB obtained by averaging X-Trans mosaics over 6×6 cells, with no demosaicer involved.

## Dataset

The network is trained on synthetic input/target pairs generated from real RAW photos. The build process works as follows:

1. **Ground truth generation** (`build_dataset.py`): RAW files (RAF, ARW, CR2, etc.) are demosaiced with LibRaw — Markesteijn three-pass for X-Trans, AHD for Bayer — in linear sensor space with no white balance or colour correction, scaled so that the sensor's white level is 65535, and downscaled 2× by area averaging to produce reference images stored as uint16 `.npy` files. The builder writes a `build_info.json` into the dataset directory (code revision and options); it refuses to add to a directory built differently, and `train.py` refuses a directory without the record.

2. **Synthetic re-mosaicing**: during training, patches are cropped at any offset from the ground truth and re-mosaiced through the sensor's pattern to form the network's input, so the model learns from a clean demosaic "re-captured" through the CFA.

3. **Augmentations**: random flips and 90° rotations, a further 2× shrink of most patches, optional Poisson-Gaussian noise (off in the S configuration: with it the model learned to denoise, which flattened faint texture and smeared dark areas), optional OLPF (anti-aliasing filter) blur, and a random gain of up to ±3 stops inside the model's normalisation. Targets and mosaics are clamped at the clip level.

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

Set up the repository as described above, then run the desktop app from the repository root:

```bash
npm run build --workspace desktop
npm run start --workspace desktop
```

Build a local installer with `npm run dist --workspace desktop -- --mac` on macOS or `npm run dist --workspace desktop -- --win` on Windows. The macOS build produces an arm64 DMG; the Windows build produces an x64 NSIS installer. These local builds are unsigned for distribution. On macOS, if the first launch is blocked, use System Settings → Privacy & Security → **Open Anyway**. On Windows, if SmartScreen appears, select **More info** → **Run anyway**.

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
