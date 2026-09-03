# What-veon?

X-veon: neural network demosaicing for Bayer and X-Trans sensors. 

This project consists of two parts: first one is the neural net itself with a bunch of scripts for dataset building and training, the other is a web application with a full RAW development pipeline.

## Neural network

The demosaicing model is a U-Net (encoder-decoder with skip connections) that takes a 4-channel input — the raw CFA mosaic value plus 3 binary masks indicating which color filter covers each pixel — and outputs a full-color 3-channel RGB image.

The encoder has 4 downsampling stages (64 → 128 → 256 → 512 → 1024 channels at the bottleneck for full-width model), each consisting of two 3×3 convolutions with BatchNorm and ReLU, followed by 2×2 max pooling. The decoder mirrors this with transposed convolutions for upsampling and skip connections from the corresponding encoder stage.

A key design choice is the residual CFA skip: the single-channel mosaic value is broadcast to all 3 output channels as a baseline, and the network only learns the color correction deltas on top of it. This makes the model largely exposure-agnostic — it doesn't need to reproduce absolute brightness, just fill in the missing color information.

The architecture is fully CFA-agnostic: the same model currently works for both 6×6 X-Trans and 2×2 Bayer patterns. It should be trivial to add Quad HDR support if necessary.

## Dataset

The network is trained on synthetic input/target pairs generated from real RAW photos. The build process works as follows:

1. **Ground truth generation**: RAW files (RAF, ARW, CR2, etc.) are demosaiced using traditional algorithms — DHT for X-Trans, AHD for Bayer — in linear sensor space with no white balance or color correction applied. The results are downscaled 4x via area averaging to produce clean, alias-free reference images stored as float32 `.npy` files.

2. **Synthetic re-mosaicing**: During training, patches are randomly cropped from the ground truth and re-mosaiced through the appropriate CFA pattern to create the network's input. This means the model never sees the original noisy RAW data — it learns from a clean demosaic that has been "re-captured" through the CFA.

3. **Augmentations**: Each patch gets random flips, additive Gaussian noise, exposure shifts (pushing toward clipping), white balance perturbation in log space, and optional OLPF (anti-aliasing filter) blur simulation. These help the model generalize across cameras and shooting conditions.

4. **Torture patterns**: A fraction of synthetic gradient and edge patterns can be mixed into the training set to improve performance on worst-case inputs like fine diagonal lines and color fringes near Nyquist.

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

`npm run build` and `npx vite preview` need `XV_CHANNEL` (`stable`, `beta` or `dev`); see `RELEASING.md`.

## License

This project uses a multi-license structure:

| Component | License | SPDX Identifier |
|---|---|---|
| Neural network code (model, training, losses, dataset) | MIT | `MIT` |
| Trained model weights (`checkpoints_*/`) | Creative Commons Attribution 4.0 | `CC-BY-4.0` |
| Processing pipeline, web app, and everything else | GNU GPL v3 or later | `GPL-3.0-or-later` |

See [LICENSE](LICENSE) for details and [LICENSES/](LICENSES/) for full license texts.

## Acknowledgments

Parts of the code were adapted from various open-source projects:
- darktable (segmentation-based highlight reconstruction, reference image pipeline)
- Jed Smith's OpenDRT and ART CTL by agriggio (tone mapping)
