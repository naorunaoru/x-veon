# What-veon?

X-veon: neural network demosaicing for Bayer and X-Trans sensors. 

This project consists of two parts: first one is the neural net itself with a bunch of scripts for dataset building and training, the other is a web application with a full RAW development pipeline.

## Neural network

The demosaicing model is a U-Net-style encoder-decoder with skip connections. At inference time it takes the raw CFA mosaic as input plus per-image white-balance coefficients. CFA masks and a per-pixel WB mask are generated internally from the stored CFA pattern, and the network outputs a full-color 3-channel RGB image.

The current model family uses four downsampling stages with configurable `base_width` (the default deployed width is 16). The blocks use convolution + GroupNorm + ReLU, and the decoder mirrors the encoder with upsampling and skip connections. For non-Bayer patterns, sinusoidal positional encodings are injected so the same architecture can support both 2×2 Bayer and 6×6 X-Trans layouts.

A key design choice is the residual CFA skip: the single-channel mosaic value is copied into its sampled color channel as a baseline, and the network learns the missing color reconstruction and correction deltas on top of it.

The architecture is CFA-agnostic and currently supports both Bayer and X-Trans with the same model family.

## Dataset

The network is trained on synthetic input/target pairs generated from real RAW photos. The build process works as follows:

1. **Ground truth generation**: RAW files (RAF, ARW, CR2, etc.) are demosaiced using traditional algorithms — DHT for X-Trans, AHD for Bayer — in linear sensor space with no white balance or color correction applied. The results are downscaled 2x via area averaging and stored as `.npy` training targets with metadata sidecars.

2. **Synthetic re-mosaicing**: During training, patches are randomly cropped from the ground truth and re-mosaiced through the appropriate CFA pattern to create the network's input. The model learns on synthetic clean mosaics rather than directly on the original noisy RAW samples.

3. **Augmentations**: Training patches can receive random flips, additive Gaussian/read+shot noise, white-balance perturbation, optional OLPF blur simulation, bright-spot augmentation, and optional 2x crop/downscale augmentation.

4. **Torture patterns**: A fraction of synthetic gradient and edge patterns can be mixed into the training set to improve performance on worst-case inputs like fine diagonal lines, zippering, and color fringes near Nyquist.

## Web application

A small, fully offline (as in all processing is done in the browser) web application was built along the model. It uses ONNX WebGPU runtime for inference, so a decent GPU is required. Processing times on an M1 Macbook Pro are in the tens of seconds at worst.

### Live demo:

https://naorunaoru.github.io/x-veon

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

## Checkpoints

Checkpoint versioning and promotion policy are documented in [CHECKPOINT_POLICY.md](CHECKPOINT_POLICY.md).

Current supported baseline:
- **v6.1.4**
- base width **16** is the default model size
- browser exports and registry are pruned to the current baseline family

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
