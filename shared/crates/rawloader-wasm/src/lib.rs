use wasm_bindgen::prelude::*;
use js_sys;

mod exif_parse;

fn usize_to_js_u16(arr: &[usize]) -> js_sys::Uint16Array {
    let jsarr = js_sys::Uint16Array::new_with_length(arr.len() as u32);
    for (i, x) in arr.iter().enumerate() {
        jsarr.set_index(i as u32, *x as u16);
    }
    jsarr
}

fn u16_to_js_u16(arr: &[u16]) -> js_sys::Uint16Array {
    let jsarr = js_sys::Uint16Array::new_with_length(arr.len() as u32);
    for (i, x) in arr.iter().enumerate() {
        jsarr.set_index(i as u32, *x);
    }
    jsarr
}

fn f32_to_js_f32(arr: &[f32]) -> js_sys::Float32Array {
    let jsarr = js_sys::Float32Array::new_with_length(arr.len() as u32);
    for (i, x) in arr.iter().enumerate() {
        jsarr.set_index(i as u32, *x);
    }
    jsarr
}

fn flatten_matrix<const N: usize, const M: usize>(matrix: &[[f32; M]; N]) -> js_sys::Float32Array {
    let jsarr = js_sys::Float32Array::new_with_length((N * M) as u32);
    for (i, row) in matrix.iter().enumerate() {
        for (j, x) in row.iter().enumerate() {
            jsarr.set_index((i * M + j) as u32, *x);
        }
    }
    jsarr
}

#[wasm_bindgen]
pub struct Image {
    data: js_sys::Uint16Array,
    make: String,
    model: String,
    width: usize,
    height: usize,
    cpp: usize,
    crops: js_sys::Uint16Array,
    cfastr: String,
    cfawidth: usize,
    cfaheight: usize,
    wb_coeffs: js_sys::Float32Array,
    whitelevels: js_sys::Uint16Array,
    calibrated_whitelevels: js_sys::Uint16Array,
    blacklevels: js_sys::Uint16Array,
    orientation: String,
    xyz_to_cam: js_sys::Float32Array,
    cam_to_xyz: js_sys::Float32Array,
    dr_gain: f32,
    exposure_bias: f32,
    lens_model: String,
    focal_length: f32,
    f_number: f32,
    fuji_lens_distortion: js_sys::Float32Array,
    fuji_lens_ca: js_sys::Float32Array,
    fuji_lens_vignetting: js_sys::Float32Array,
    fuji_lens_frame: js_sys::Uint16Array,
}

#[wasm_bindgen]
impl Image {
    pub fn get_data(&self) -> js_sys::Uint16Array {
        self.data.clone()
    }

    pub fn get_make(&self) -> String {
        self.make.clone()
    }

    pub fn get_model(&self) -> String {
        self.model.clone()
    }

    pub fn get_width(&self) -> usize {
        self.width
    }

    pub fn get_height(&self) -> usize {
        self.height
    }

    pub fn get_cpp(&self) -> usize {
        self.cpp
    }

    pub fn get_crops(&self) -> js_sys::Uint16Array {
        self.crops.clone()
    }

    pub fn get_cfastr(&self) -> String {
        self.cfastr.clone()
    }

    pub fn get_cfawidth(&self) -> usize {
        self.cfawidth
    }

    pub fn get_cfaheight(&self) -> usize {
        self.cfaheight
    }

    pub fn get_wb_coeffs(&self) -> js_sys::Float32Array {
        self.wb_coeffs.clone()
    }

    pub fn get_whitelevels(&self) -> js_sys::Uint16Array {
        self.whitelevels.clone()
    }

    /// The white levels measured from the data where the sensor clips below the metadata level
    /// (RGBE order, like `get_whitelevels`).
    pub fn get_calibrated_whitelevels(&self) -> js_sys::Uint16Array {
        self.calibrated_whitelevels.clone()
    }

    pub fn get_blacklevels(&self) -> js_sys::Uint16Array {
        self.blacklevels.clone()
    }

    pub fn get_orientation(&self) -> String {
        self.orientation.clone()
    }

    pub fn get_xyz_to_cam(&self) -> js_sys::Float32Array {
        self.xyz_to_cam.clone()
    }

    pub fn get_cam_to_xyz(&self) -> js_sys::Float32Array {
        self.cam_to_xyz.clone()
    }

    pub fn get_dr_gain(&self) -> f32 {
        self.dr_gain
    }

    pub fn get_exposure_bias(&self) -> f32 {
        self.exposure_bias
    }

    pub fn get_lens_model(&self) -> String {
        self.lens_model.clone()
    }

    pub fn get_focal_length(&self) -> f32 {
        self.focal_length
    }

    pub fn get_f_number(&self) -> f32 {
        self.f_number
    }

    /// The camera's distortion table (Fuji RAF). It and the CA and vignetting tables are empty
    /// when the file has none.
    pub fn get_fuji_lens_distortion(&self) -> js_sys::Float32Array {
        self.fuji_lens_distortion.clone()
    }

    pub fn get_fuji_lens_ca(&self) -> js_sys::Float32Array {
        self.fuji_lens_ca.clone()
    }

    pub fn get_fuji_lens_vignetting(&self) -> js_sys::Float32Array {
        self.fuji_lens_vignetting.clone()
    }

    /// [full height, full width, crop top, crop left, cropped height, cropped width], or empty
    /// when the file has no tables.
    pub fn get_fuji_lens_frame(&self) -> js_sys::Uint16Array {
        self.fuji_lens_frame.clone()
    }
}

#[wasm_bindgen]
pub fn decode_image(arr: js_sys::Uint8Array) -> Result<Image, JsValue> {
    console_error_panic_hook::set_once();
    let vec = arr.to_vec();
    let image = rawloader::decode(&mut std::io::Cursor::new(&vec[..]))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    to_image(image, &vec)
}

/// A decode in steps, so that a compressed Fuji RAF can be decoded strip by strip: its strips
/// decode independently. The caller cuts each strip's bytes out of its own copy of the file
/// (`layout`), has them decoded in other instances of this module (`decode_fuji_strip`, one
/// worker per strip), hands every block back with `put`, and gets the `Image` with `finish`.
/// Any other file has no layout, and `finish` alone decodes it, like `decode_image`.
#[wasm_bindgen]
pub struct RawDecode {
    file: Vec<u8>,
    buffer: rawloader::Buffer,
    strips: Option<rawloader::FujiStrips>,
    pixels: Vec<u16>,
    filled: Vec<bool>,
}

#[wasm_bindgen]
impl RawDecode {
    pub fn open(arr: js_sys::Uint8Array) -> Result<RawDecode, JsValue> {
        console_error_panic_hook::set_once();
        let file = arr.to_vec();
        let buffer = rawloader::Buffer::new(&mut std::io::Cursor::new(&file[..])).map_err(|e| JsValue::from_str(&e))?;
        // A file whose layout can't be read has no strips: `finish` reports what is wrong with
        // it. Nor has one whose strips reach past the end of the file, since the caller cuts
        // them from the file itself, without the decoder's padding.
        let within = |end: usize| end <= file.len();
        let strips = rawloader::fuji_strips(&buffer).ok().flatten()
            .filter(|f| within(f.header + 16) && f.strips.iter().all(|s| within(s.data.end) && within(s.q_bases.end)));
        let (pixels, filled) = match &strips {
            Some(f) => (vec![0u16; f.width * f.height], vec![false; f.strips.len()]),
            None => (Vec::new(), Vec::new()),
        };
        Ok(RawDecode { file, buffer, strips, pixels, filled })
    }

    /// Empty unless the file is decoded in strips. Otherwise, with offsets into the file:
    /// [header (16 bytes), image height, then for each strip its data start and end, its
    /// q_bases start and end, its coded width and its column count].
    pub fn layout(&self) -> js_sys::Uint32Array {
        let mut layout = Vec::new();
        if let Some(f) = &self.strips {
            layout.extend([f.header as u32, f.height as u32]);
            for s in &f.strips {
                layout.extend([s.data.start, s.data.end, s.q_bases.start, s.q_bases.end, s.strip_width, s.cols].map(|v| v as u32));
            }
        }
        js_sys::Uint32Array::from(&layout[..])
    }

    /// Copies strip `index`'s decoded block (its columns, row by row) into the image.
    pub fn put(&mut self, index: usize, block: &[u16]) -> Result<(), JsValue> {
        let strips = self.strips.as_ref().ok_or_else(|| JsValue::from_str("Not a file decoded in strips"))?;
        let strip = strips.strips.get(index).ok_or_else(|| JsValue::from_str("No such strip"))?;
        if block.len() != strip.cols * strips.height {
            return Err(JsValue::from_str("A strip's block doesn't match its size"));
        }
        if strip.cols > 0 {
            for (row, src) in self.pixels.chunks_exact_mut(strips.width).zip(block.chunks_exact(strip.cols)) {
                row[strip.x..strip.x + strip.cols].copy_from_slice(src);
            }
        }
        self.filled[index] = true;
        Ok(())
    }

    /// The decoded image: from the strips, once every one has been put, or by decoding a file
    /// that has none.
    pub fn finish(self) -> Result<Image, JsValue> {
        let image = if self.strips.is_none() {
            rawloader::decode_buffer(&self.buffer)
        } else if self.filled.iter().all(|&f| f) {
            rawloader::decode_with_fuji_pixels(&self.buffer, self.pixels)
        } else {
            return Err(JsValue::from_str("Not every strip was decoded"));
        };
        to_image(image.map_err(|e| JsValue::from_str(&e.to_string()))?, &self.file)
    }
}

/// Decodes one strip of a compressed Fuji RAF into a block of its columns, row by row. `header`
/// is the 16 bytes at the layout's header offset; the rest are the strip's bytes and numbers
/// from `RawDecode.layout`. Stateless, and it checks its arguments.
#[wasm_bindgen]
pub fn decode_fuji_strip(header: &[u8], data: &[u8], q_bases: &[u8], strip_width: usize, cols: usize, height: usize) -> Result<Vec<u16>, JsValue> {
    console_error_panic_hook::set_once();
    rawloader::decode_fuji_strip(header, data, q_bases, strip_width, cols, height)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

fn to_image(image: rawloader::RawImage, vec: &[u8]) -> Result<Image, JsValue> {
    let cam_to_xyz = image.cam_to_xyz();
    let calibrated_whitelevels = image.calibrated_whitelevels();

    let vector = match image.data {
        rawloader::RawImageData::Integer(v) => v,
        _ => return Err(JsValue::from_str("Float raw data is not supported")),
    };

    let orientation_string = match image.orientation {
        rawloader::Orientation::Normal => "Normal",
        rawloader::Orientation::HorizontalFlip => "HorizontalFlip",
        rawloader::Orientation::Rotate180 => "Rotate180",
        rawloader::Orientation::VerticalFlip => "VerticalFlip",
        rawloader::Orientation::Transpose => "Transpose",
        rawloader::Orientation::Rotate90 => "Rotate90",
        rawloader::Orientation::Transverse => "Transverse",
        rawloader::Orientation::Rotate270 => "Rotate270",
        rawloader::Orientation::Unknown => "Unknown",
    };

    let dr_gain = exif_parse::extract_dr_gain(vec);
    let exif_meta = exif_parse::extract_exif_meta(vec);
    let lens = image.fuji_lens.as_ref();

    Ok(Image {
        make: image.make.clone(),
        model: image.model.clone(),
        width: image.width,
        height: image.height,
        cpp: image.cpp,
        crops: usize_to_js_u16(&image.crops),
        cfastr: image.cfa.name.clone(),
        cfawidth: image.cfa.width,
        cfaheight: image.cfa.height,
        wb_coeffs: f32_to_js_f32(&image.wb_coeffs),
        whitelevels: u16_to_js_u16(&image.whitelevels),
        calibrated_whitelevels: u16_to_js_u16(&calibrated_whitelevels),
        blacklevels: u16_to_js_u16(&image.blacklevels),
        orientation: orientation_string.to_owned(),
        xyz_to_cam: flatten_matrix(&image.xyz_to_cam),
        cam_to_xyz: flatten_matrix(&cam_to_xyz),
        dr_gain,
        exposure_bias: exif_meta.exposure_bias,
        lens_model: exif_meta.lens_model,
        focal_length: exif_meta.focal_length,
        f_number: exif_meta.f_number,
        fuji_lens_distortion: f32_to_js_f32(lens.map_or(&[], |l| &l.distortion[..])),
        fuji_lens_ca: f32_to_js_f32(lens.map_or(&[], |l| &l.ca[..])),
        fuji_lens_vignetting: f32_to_js_f32(lens.map_or(&[], |l| &l.vignetting[..])),
        fuji_lens_frame: u16_to_js_u16(lens.map_or(&[], |l| &l.frame[..])),
        data: js_sys::Uint16Array::from(&vector[..]),
    })
}
