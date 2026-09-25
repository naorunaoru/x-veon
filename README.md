# What-veon?

X-veon: neural network demosaicing for Bayer and X-Trans sensors. 

This project consists of two parts: first one is the neural net itself with a bunch of scripts for dataset building and training, the other is a web application with a full RAW development pipeline.

## Neural network

The demosaicing model is a U-Net (encoder-decoder with skip connections, `model.py`) with a 5-channel input: the raw CFA mosaic value, 3 binary masks marking which colour filter covers each pixel, and a clip-proximity channel (0 below half of the clip level, ramping to 1 at clipping). It outputs a full-colour 3-channel image in camera RGB, without white balance.

The encoder has 4 downsampling stages (strided convolutions; channel widths `base_width × 1, 2, 4, 8, 16`). Each stage is two convolutions with GroupNorm and ReLU; the decoder upsamples with 1×1 convolutions and PixelShuffle and concatenates the matching encoder stage. For X-Trans the first convolution is 7×7 and the input also carries sin/cos encodings of the 6×6 CFA phase.

A key design choice is the residual CFA skip: each photosite's value is placed in its own colour channel as a baseline (`cfa × masks`), and the network learns the missing colours on top of it. This keeps the model largely exposure-agnostic.

The same architecture serves both 6×6 X-Trans and 2×2 Bayer patterns, with a separate model per sensor type. The models shipped in `web/public/checkpoints/` were exported before the current architecture (max-pool/transposed-convolution, no normalisation), so the current `model.py` cannot load them; retrain to reproduce them.

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

The RAW decoder is a git submodule (`web/wasm/vendor/rawloader`, a pinned fork of rawloader). Clone with `git clone --recurse-submodules`, or run `git submodule update --init` in an existing checkout. Then:

```
cd web
npm run setup          # rust wasm target, npm install, wasm builds
npm run build:lensfun  # lens-correction data (gitignored, needed once)
npm run dev
```

`npm run dev` serves without cross-origin isolation, like GitHub Pages.

`npm run build` and `npx vite preview` need `XV_CHANNEL` (`stable`, `beta` or `dev`); see `RELEASING.md`.

Source layout (`web/src`): `lib` (shared types and the method/format catalogue, no dependencies) ← `gpu` (the shared WebGPU device) ← `pipeline` (RAW → processed image, framework-free) and `renderer` (display, histogram, readback) ← `app` (the store as pure slices, persistence/library/processing/export/bootstrap services, hooks) ← `components` (React UI). Imports only go left; `src/test/layers.test.ts` fails the suite on a violation and names the file and line.

## License

This project uses a multi-license structure:

| Component | License | SPDX Identifier |
|---|---|---|
| Neural network code (model, training, losses, dataset) | MIT | `MIT` |
| Trained model weights and ONNX checkpoints (`web/public/checkpoints/`) | Creative Commons Attribution 4.0 | `CC-BY-4.0` |
| Processing pipeline, web app, and everything else | GNU GPL v3 or later | `GPL-3.0-or-later` |

See [LICENSE](LICENSE) for details and [LICENSES/](LICENSES/) for full license texts.

## Acknowledgments

RAW decoding uses a fork of [rawloader](https://github.com/pedrocr/rawloader) (LGPL-2.1), included as the `web/wasm/vendor/rawloader` submodule.

Parts of the code were adapted from various open-source projects:
- darktable (segmentation-based highlight reconstruction, reference image pipeline)
- Jed Smith's OpenDRT and ART CTL by agriggio (tone mapping)
