//! Library to extract the raw data and some metadata from digital camera
//! images. Given an image in a supported format and camera you will be able to get
//! everything needed to process the image
//!
//! # Example
//! ```rust,no_run
//! use std::env;
//! use std::fs::File;
//! use std::io::prelude::*;
//! use std::io::BufWriter;
//!
//! fn main() {
//!   let args: Vec<_> = env::args().collect();
//!   if args.len() != 2 {
//!     println!("Usage: {} <file>", args[0]);
//!     std::process::exit(2);
//!   }
//!   let file = &args[1];
//!   let image = rawloader::decode_file(file).unwrap();
//!
//!   // Write out the image as a grayscale PPM
//!   let mut f = BufWriter::new(File::create(format!("{}.ppm",file)).unwrap());
//!   let preamble = format!("P6 {} {} {}\n", image.width, image.height, 65535).into_bytes();
//!   f.write_all(&preamble).unwrap();
//!   if let rawloader::RawImageData::Integer(data) = image.data {
//!     for pix in data {
//!       // Do an extremely crude "demosaic" by setting R=G=B
//!       let pixhigh = (pix>>8) as u8;
//!       let pixlow  = (pix&0x0f) as u8;
//!       f.write_all(&[pixhigh, pixlow, pixhigh, pixlow, pixhigh, pixlow]).unwrap()
//!     }
//!   } else {
//!     eprintln!("Don't know how to process non-integer raw files");
//!   }
//! }
//! ```

// #![deny(
//   missing_docs,
//   missing_debug_implementations,
//   missing_copy_implementations,
//   unsafe_code,
//   unstable_features,
//   unused_import_braces,
//   unused_qualifications
// )]

use lazy_static::lazy_static;
use wasm_bindgen::prelude::*;
use js_sys;
extern crate console_error_panic_hook;
mod decoders;
pub use decoders::{RawImage, RawImageData, Orientation, cfa::CFA, Encoding};
#[doc(hidden)] pub use decoders::Buffer;
#[doc(hidden)] pub use decoders::RawLoader;

lazy_static! {
  static ref LOADER: RawLoader = decoders::RawLoader::new();
}

use std::path::Path;
use std::error::Error;
use std::fmt;
use std::io::Read;

#[wasm_bindgen]
pub struct Image {
    data: js_sys::Uint16Array,
    original: js_sys::Uint8Array,
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
    bps: usize,
    offset: usize,
    encoding: Encoding
}

#[wasm_bindgen]
impl Image{
    pub fn get_data(&self) -> js_sys::Uint16Array {
        self.data.clone()
    }

    pub fn get_original(&self) -> js_sys::Uint8Array {
        self.original.clone()
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

    pub fn get_bps(&self) -> usize {
      self.bps
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

    pub fn encode(&self, image: js_sys::Uint16Array) -> js_sys::Uint8Array {
      let image_vec: Vec<u16> = image.to_vec();
      // panic!("{}", image_vec.len());
      let vector : Vec<u8> = match LOADER.encode(self.original.to_vec(), image_vec, self.bps, self.offset, self.encoding) {
        Ok(value) => value,
        Err(e) => panic!("{}", e)
      };
      js_sys::Uint8Array::from(&vector[..])
    }
}

pub fn usize_to_js_u16(arr: &[usize]) -> js_sys::Uint16Array {
  let jsarr = js_sys::Uint16Array::new_with_length(arr.len() as u32);
  for (i, x) in arr.iter().enumerate() {
      jsarr.set_index(i as u32, x.clone() as u16)
  }
  jsarr
}

pub fn u16_to_js_u16(arr: &[u16]) -> js_sys::Uint16Array {
  let jsarr = js_sys::Uint16Array::new_with_length(arr.len() as u32);
  for (i, x) in arr.iter().enumerate() {
    jsarr.set_index(i as u32, x.clone() as u16)
  }
  jsarr
}

pub fn f32_to_js_f32(arr: &[f32]) -> js_sys::Float32Array {
  let jsarr = js_sys::Float32Array::new_with_length(arr.len() as u32);
  for (i, x) in arr.iter().enumerate() {
    jsarr.set_index(i as u32, x.clone() as f32)
  }
  jsarr
}

pub fn flatten_matrix<const N: usize, const M: usize>(matrix: &[[f32;M];N]) -> js_sys::Float32Array {
  let jsarr = js_sys::Float32Array::new_with_length((N*M) as u32);
  for (i, row) in matrix.iter().enumerate() {
    for (j, x) in row.iter().enumerate() {
      jsarr.set_index((i*M + j) as u32, x.clone() as f32)
    }
  }
  jsarr
}

#[wasm_bindgen]
pub fn decode_image(arr: js_sys::Uint8Array) -> Image{
  console_error_panic_hook::set_once();
  let vec = &arr.to_vec();
  let image = decode_file_vec(vec).unwrap();
  let m = &image.cam_to_xyz().clone();
  let vector = match image.data {
      RawImageData::Integer(vec) => vec,
      _ => panic!("cannot decode floats yet")
    };


  let orientationString = match image.orientation {
    decoders::Orientation::Normal => "Normal",
    decoders::Orientation::HorizontalFlip => "HorizontalFlip",
    decoders::Orientation::Rotate180 => "Rotate180",
    decoders::Orientation::VerticalFlip => "VerticalFlip",
    decoders::Orientation::Transpose => "Transpose",
    decoders::Orientation::Rotate90 => "Rotate90",
    decoders::Orientation::Transverse => "Transverse",
    decoders::Orientation::Rotate270 => "Rotate270",
    decoders::Orientation::Unknown => "Unknown",
  };

  let result = Image {
      make: image.make.clone(),
      model: image.model.clone(),
      width: image.width.clone(),
      height: image.height.clone(),
      cpp: image.cpp.clone(),
      crops: usize_to_js_u16(&image.crops),
      cfastr: image.cfa.name.clone(),
      cfawidth: image.cfa.width.clone(),
      cfaheight: image.cfa.height.clone(),
      wb_coeffs: f32_to_js_f32(&image.wb_coeffs),
      whitelevels: u16_to_js_u16(&image.whitelevels),
      blacklevels: u16_to_js_u16(&image.blacklevels),
      orientation: orientationString.to_owned(),
      xyz_to_cam: flatten_matrix(&image.xyz_to_cam),
      cam_to_xyz: flatten_matrix(m),
      bps: image.bps,
      offset: image.offset,
      encoding: image.encoding,
      data: js_sys::Uint16Array::from(&vector[..]),
      original: arr
  };
  result
}

/// Decode array
pub fn decode_file_vec(vec: &Vec<u8>) -> Result<RawImage,RawLoaderError> {
  LOADER.decode_file_vec(vec).map_err(|err| RawLoaderError::new(err))
}

/// Error type for any reason for the decode to fail
#[derive(Debug)]
pub struct RawLoaderError {
  msg: String,
}

impl fmt::Display for RawLoaderError {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "RawLoaderError: \"{}\"", self.msg)
  }
}

impl Error for RawLoaderError {
  // Implement description so that older versions of rust still work
  fn description(&self) -> &str {
    "description() is deprecated; use Display"
  }
}

impl RawLoaderError {
  fn new(msg: String) -> Self {
    Self {
      msg,
    }
  }
}

/// Take a path to a raw file and return a decoded image or an error
///
/// # Example
/// ```rust,ignore
/// let image = match rawloader::decode_file("path/to/your/file.RAW") {
///   Ok(val) => val,
///   Err(e) => ... some appropriate action when the file is unreadable ...
/// };
/// ```
pub fn decode_file<P: AsRef<Path>>(path: P) -> Result<RawImage,RawLoaderError> {
  LOADER.decode_file(path.as_ref()).map_err(|err| RawLoaderError::new(err))
}

/// Take a readable source and return a decoded image or an error
///
/// # Example
/// ```rust,ignore
/// let mut file = match File::open(path).unwrap();
/// let image = match rawloader::decode(&mut file) {
///   Ok(val) => val,
///   Err(e) => ... some appropriate action when the file is unreadable ...
/// };
/// ```
pub fn decode(reader: &mut dyn Read) -> Result<RawImage,RawLoaderError> {
  LOADER.decode(reader, false).map_err(|err| RawLoaderError::new(err))
}

// Used to force lazy_static initializations. Useful for fuzzing.
#[doc(hidden)]
pub fn force_initialization() {
  lazy_static::initialize(&LOADER);
}

// Used for fuzzing targets that just want to test the actual decoders instead of the full formats
// with all their TIFF and other crazyness
#[doc(hidden)]
pub fn decode_unwrapped(reader: &mut dyn Read) -> Result<RawImageData,RawLoaderError> {
  LOADER.decode_unwrapped(reader).map_err(|err| RawLoaderError::new(err))
}

// Used for fuzzing everything but the decoders themselves
#[doc(hidden)]
pub fn decode_dummy(reader: &mut dyn Read) -> Result<RawImage,RawLoaderError> {
  LOADER.decode(reader, true).map_err(|err| RawLoaderError::new(err))
}
