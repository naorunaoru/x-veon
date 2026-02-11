# X-Trans Demosaic

Neural network demosaicing for Fujifilm X-Trans sensors. A U-Net trained in linear sensor space reconstructs full RGB from the 6x6 X-Trans CFA pattern, with optional white balance integration and HDR output via HLG-encoded AVIF.

## Live demo:
https://naorunaoru.github.io/x-veon

Works completely in the browser, from RAF loading to inference (ONNX WebGPU runtime) to saving.

Note that this is still early in progress.
