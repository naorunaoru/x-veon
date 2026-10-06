//! Raw values for the display that holds the app's window (spec §8). Main calls this on its own
//! thread: AppKit is main-thread only, and Electron runs main's JavaScript on that thread.
use napi::bindgen_prelude::Buffer;
use napi_derive::napi;

/// macOS fills the three EDR values; Windows fills the luminance range, the SDR white level in
/// nits and whether the output is in HDR mode. Field names match the shared `DisplayReadings`.
#[napi(object)]
#[derive(Debug, Default, Clone, PartialEq)]
pub struct DisplayReadings {
    pub current_edr: Option<f64>,
    pub potential_edr: Option<f64>,
    pub reference_edr: Option<f64>,
    pub max_luminance: Option<f64>,
    pub max_full_frame_luminance: Option<f64>,
    pub min_luminance: Option<f64>,
    pub sdr_white: Option<f64>,
    pub hdr_enabled: Option<bool>,
}

/// `handle` is Electron's `getNativeWindowHandle()`: an `NSView*` on macOS, an `HWND` on Windows.
/// Without a usable handle: the main screen (macOS) or the primary monitor (Windows).
#[napi]
pub fn display_readings(handle: Option<Buffer>) -> Option<DisplayReadings> {
    platform::read(handle.as_deref().and_then(handle_pointer))
}

/// The pointer-sized value at the start of a native handle; None for short or null handles.
fn handle_pointer(bytes: &[u8]) -> Option<usize> {
    const SIZE: usize = std::mem::size_of::<usize>();
    let value = usize::from_ne_bytes(bytes.get(..SIZE)?.try_into().ok()?);
    (value != 0).then_some(value)
}

#[cfg(target_os = "macos")]
mod platform {
    use super::DisplayReadings;
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSScreen, NSView};

    pub fn read(view: Option<usize>) -> Option<DisplayReadings> {
        let mtm = MainThreadMarker::new()?;
        let screen = view
            // SAFETY: main passes its own live window's NSView, on the main thread.
            .and_then(|pointer| unsafe { (pointer as *const NSView).as_ref() })
            .and_then(|view| view.window())
            .and_then(|window| window.screen())
            .or_else(|| NSScreen::mainScreen(mtm))?;
        Some(DisplayReadings {
            current_edr: Some(screen.maximumExtendedDynamicRangeColorComponentValue()),
            potential_edr: Some(screen.maximumPotentialExtendedDynamicRangeColorComponentValue()),
            reference_edr: Some(screen.maximumReferenceExtendedDynamicRangeColorComponentValue()),
            ..Default::default()
        })
    }
}

#[cfg(windows)]
mod platform {
    use super::DisplayReadings;
    use windows::core::Interface;
    use windows::Win32::Devices::Display::{
        DisplayConfigGetDeviceInfo, GetDisplayConfigBufferSizes, QueryDisplayConfig,
        DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL, DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME,
        DISPLAYCONFIG_DEVICE_INFO_HEADER, DISPLAYCONFIG_MODE_INFO, DISPLAYCONFIG_PATH_INFO,
        DISPLAYCONFIG_SDR_WHITE_LEVEL, DISPLAYCONFIG_SOURCE_DEVICE_NAME, QDC_ONLY_ACTIVE_PATHS,
    };
    use windows::Win32::Foundation::{ERROR_SUCCESS, HWND};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1, IDXGIOutput6, DXGI_OUTPUT_DESC1};
    use windows::Win32::Graphics::Gdi::{
        GetMonitorInfoW, MonitorFromWindow, HMONITOR, MONITORINFO, MONITORINFOEXW,
        MONITOR_DEFAULTTONEAREST, MONITOR_DEFAULTTOPRIMARY,
    };

    pub fn read(hwnd: Option<usize>) -> Option<DisplayReadings> {
        let flags = if hwnd.is_some() { MONITOR_DEFAULTTONEAREST } else { MONITOR_DEFAULTTOPRIMARY };
        let monitor = unsafe { MonitorFromWindow(HWND(hwnd.unwrap_or(0) as *mut _), flags) };
        if monitor.is_invalid() { return None; }
        let desc = output_desc(monitor)?;
        Some(DisplayReadings {
            max_luminance: Some(desc.MaxLuminance as f64),
            max_full_frame_luminance: Some(desc.MaxFullFrameLuminance as f64),
            min_luminance: Some(desc.MinLuminance as f64),
            hdr_enabled: Some(desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020),
            sdr_white: sdr_white_nits(monitor),
            ..Default::default()
        })
    }

    /// The DXGI output that drives `monitor`.
    fn output_desc(monitor: HMONITOR) -> Option<DXGI_OUTPUT_DESC1> {
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.ok()?;
        for a in 0.. {
            let Ok(adapter) = (unsafe { factory.EnumAdapters1(a) }) else { break };
            for o in 0.. {
                let Ok(output) = (unsafe { adapter.EnumOutputs(o) }) else { break };
                let Ok(desc) = (unsafe { output.GetDesc() }) else { continue };
                if desc.Monitor == monitor {
                    return unsafe { output.cast::<IDXGIOutput6>().ok()?.GetDesc1() }.ok();
                }
            }
        }
        None
    }

    /// Windows reports SDR white in thousandths of 80 nits.
    fn sdr_white_nits(monitor: HMONITOR) -> Option<f64> {
        let mut info = MONITORINFOEXW::default();
        info.monitorInfo.cbSize = std::mem::size_of::<MONITORINFOEXW>() as u32;
        if !unsafe { GetMonitorInfoW(monitor, &mut info.monitorInfo as *mut MONITORINFO) }.as_bool() { return None; }
        let (mut path_count, mut mode_count) = (0u32, 0u32);
        if unsafe { GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &mut path_count, &mut mode_count) } != ERROR_SUCCESS { return None; }
        let mut paths = vec![DISPLAYCONFIG_PATH_INFO::default(); path_count as usize];
        let mut modes = vec![DISPLAYCONFIG_MODE_INFO::default(); mode_count as usize];
        if unsafe { QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &mut path_count, paths.as_mut_ptr(), &mut mode_count, modes.as_mut_ptr(), None) } != ERROR_SUCCESS { return None; }
        for path in &paths[..path_count as usize] {
            let mut source = DISPLAYCONFIG_SOURCE_DEVICE_NAME::default();
            source.header = DISPLAYCONFIG_DEVICE_INFO_HEADER {
                r#type: DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME,
                size: std::mem::size_of::<DISPLAYCONFIG_SOURCE_DEVICE_NAME>() as u32,
                adapterId: path.sourceInfo.adapterId,
                id: path.sourceInfo.id,
            };
            if unsafe { DisplayConfigGetDeviceInfo(&mut source.header) } != 0 { continue; }
            if !same_name(&source.viewGdiDeviceName, &info.szDevice) { continue; }
            let mut white = DISPLAYCONFIG_SDR_WHITE_LEVEL {
                header: DISPLAYCONFIG_DEVICE_INFO_HEADER {
                    r#type: DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL,
                    size: std::mem::size_of::<DISPLAYCONFIG_SDR_WHITE_LEVEL>() as u32,
                    adapterId: path.targetInfo.adapterId,
                    id: path.targetInfo.id,
                },
                SDRWhiteLevel: 0,
            };
            if unsafe { DisplayConfigGetDeviceInfo(&mut white.header) } != 0 || white.SDRWhiteLevel == 0 { return None; }
            return Some(white.SDRWhiteLevel as f64 / 1000.0 * 80.0);
        }
        None
    }

    fn same_name(a: &[u16; 32], b: &[u16; 32]) -> bool {
        let end = |s: &[u16; 32]| s.iter().position(|&c| c == 0).unwrap_or(s.len());
        a[..end(a)] == b[..end(b)]
    }
}

#[cfg(not(any(target_os = "macos", windows)))]
mod platform {
    pub fn read(_: Option<usize>) -> Option<super::DisplayReadings> { None }
}
