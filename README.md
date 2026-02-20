# What-veon?

X-veon: neural network demosaicing for Bayer and X-Trans sensors. 

Comparison images: https://naorunaoru.github.io/x-veon/comparison.html

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

## Something about giants and their shoulders. Mom, I'm on TV

Parts of the code were adapted piecemeal from various open-source projects, to name a few: 
- darktable (segmentation-based highlight reconstruction, reference image pipeline)
- Jed Smith's OpenDRT and ART CTL by agriggio (tone mapping)
