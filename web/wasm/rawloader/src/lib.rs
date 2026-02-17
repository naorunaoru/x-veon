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
    blacklevels: js_sys::Uint16Array,
    orientation: String,
    xyz_to_cam: js_sys::Float32Array,
    cam_to_xyz: js_sys::Float32Array,
    dr_gain: f32,
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
}

#[wasm_bindgen]
pub fn decode_image(arr: js_sys::Uint8Array) -> Result<Image, JsValue> {
    console_error_panic_hook::set_once();
    let vec = arr.to_vec();
    let image = rawloader::decode_file_vec(&vec)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let cam_to_xyz = image.cam_to_xyz();

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

    let dr_gain = exif_parse::extract_dr_gain(&vec);

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
        blacklevels: u16_to_js_u16(&image.blacklevels),
        orientation: orientation_string.to_owned(),
        xyz_to_cam: flatten_matrix(&image.xyz_to_cam),
        cam_to_xyz: flatten_matrix(&cam_to_xyz),
        dr_gain,
        data: js_sys::Uint16Array::from(&vector[..]),
    })
}
