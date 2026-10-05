use napi::bindgen_prelude::*;
use napi_derive::napi;
use xveon_encode::{encode as encode_core, EncodeOptions, Format};

#[napi(object)]
pub struct EncodeInput {
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub orientation: String,
    pub quality: u32,
    pub peak_luminance: f64,
    pub threads: Option<u32>,
}

pub struct EncodeTask {
    data: Float32Array,
    hdr: Option<Float32Array>,
    input: EncodeInput,
}
impl Task for EncodeTask {
    type Output = Vec<u8>;
    type JsValue = Buffer;
    fn compute(&mut self) -> Result<Vec<u8>> {
        let format = Format::parse(&self.input.format).map_err(Error::from_reason)?;
        let threads = match self.input.threads {
            Some(n) if n > 0 => n as usize,
            _ => std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        };
        let hdr: &[f32] = self.hdr.as_deref().unwrap_or(&[]);
        encode_core(
            &self.data,
            hdr,
            self.input.width,
            self.input.height,
            &self.input.orientation,
            format,
            self.input.quality.clamp(1, 100) as u8,
            self.input.peak_luminance as f32,
            EncodeOptions { threads },
        )
        .map_err(Error::from_reason)
    }
    fn resolve(&mut self, _env: Env, output: Vec<u8>) -> Result<Buffer> {
        Ok(output.into())
    }
}
/// Runs on a libuv worker thread; rav1e spreads the frame's tiles over its own pool.
#[napi]
pub fn encode(
    data: Float32Array,
    hdr: Option<Float32Array>,
    input: EncodeInput,
) -> AsyncTask<EncodeTask> {
    AsyncTask::new(EncodeTask { data, hdr, input })
}
