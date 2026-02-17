use std::io::{BufReader, Cursor};

macro_rules! log {
    ($($arg:tt)*) => {
        web_sys::console::log_1(&format!($($arg)*).into());
    };
}

/// Extract EXIF ExposureBiasValue from raw file bytes.
/// Returns the exposure compensation in EV (e.g. -0.33, +1.0).
/// Defaults to 0.0 if tag not found.
pub fn extract_exposure_bias(raw_bytes: &[u8]) -> f32 {
    let data = if is_raf(raw_bytes) {
        let jpeg_offset = match u32::from_be_bytes(raw_bytes[84..88].try_into().unwrap_or([0;4])) as usize {
            0 => return 0.0,
            o if o >= raw_bytes.len() => return 0.0,
            o => o,
        };
        &raw_bytes[jpeg_offset..]
    } else {
        raw_bytes
    };

    let reader = std::io::BufReader::new(std::io::Cursor::new(data));
    let exif_data = match exif::Reader::new().read_from_container(&mut std::io::BufReader::new(reader)) {
        Ok(e) => e,
        Err(_) => return 0.0,
    };

    for field in exif_data.fields() {
        if field.tag == exif::Tag::ExposureBiasValue {
            if let exif::Value::SRational(vals) = &field.value {
                if let Some(r) = vals.first() {
                    if r.denom != 0 {
                        return r.num as f32 / r.denom as f32;
                    }
                }
            }
        }
    }

    0.0
}

/// Extract Fuji Dynamic Range gain from raw file bytes.
/// Returns 1.0 (DR100), 2.0 (DR200), or 4.0 (DR400).
/// Defaults to 1.0 if tag not found or not a Fuji file.
pub fn extract_dr_gain(raw_bytes: &[u8]) -> f32 {
    let dr_value = if is_raf(raw_bytes) {
        log!("[DR] RAF file detected ({} bytes)", raw_bytes.len());
        extract_dr_from_raf(raw_bytes)
    } else {
        log!("[DR] Non-RAF file ({} bytes), trying TIFF EXIF", raw_bytes.len());
        extract_dr_from_tiff(raw_bytes)
    };

    let gain = match dr_value {
        Some(200) => 2.0,
        Some(400) => 4.0,
        Some(v) => {
            log!("[DR] Tag 0x1403 = {v} → gain 1.0 (DR100 or unknown value)");
            1.0
        }
        None => {
            log!("[DR] Tag 0x1403 not found → gain 1.0");
            1.0
        }
    };

    if dr_value == Some(200) || dr_value == Some(400) {
        log!("[DR] Tag 0x1403 = {} → gain {gain}x", dr_value.unwrap());
    }

    gain
}

fn is_raf(bytes: &[u8]) -> bool {
    bytes.len() > 92 && bytes.starts_with(b"FUJIFILMCCD-RAW")
}

fn extract_dr_from_raf(bytes: &[u8]) -> Option<u32> {
    // RAF header: bytes 84-87 = offset to embedded JPEG (big-endian u32)
    let jpeg_offset = u32::from_be_bytes(bytes[84..88].try_into().ok()?) as usize;
    log!("[DR] RAF embedded JPEG at offset {jpeg_offset}");
    if jpeg_offset >= bytes.len() {
        log!("[DR] JPEG offset beyond file end");
        return None;
    }
    let jpeg_slice = &bytes[jpeg_offset..];

    // Try kamadak-exif first
    let result = parse_exif_for_dr(jpeg_slice);
    if result.is_some() {
        return result;
    }

    // Fallback to manual makernote parsing
    log!("[DR] kamadak-exif didn't find tag, trying manual makernote parse");
    parse_fuji_makernote_manual(jpeg_slice)
}

fn extract_dr_from_tiff(bytes: &[u8]) -> Option<u32> {
    parse_exif_for_dr(bytes)
}

/// Try to find Fuji DevelopmentDynamicRange (tag 0x1403) using kamadak-exif
fn parse_exif_for_dr(data: &[u8]) -> Option<u32> {
    let reader = BufReader::new(Cursor::new(data));
    let exif_data = match exif::Reader::new().read_from_container(&mut BufReader::new(reader)) {
        Ok(e) => {
            log!("[DR] kamadak-exif parsed OK, {} fields", e.fields().count());
            e
        }
        Err(e) => {
            log!("[DR] kamadak-exif parse error: {e}");
            return None;
        }
    };

    for field in exif_data.fields() {
        // Fuji DevelopmentDynamicRange = makernote tag 0x1403 (5120 decimal)
        if field.tag.number() == 0x1403 {
            log!("[DR] Found tag 0x1403 via kamadak-exif: {:?}", field.value);
            match &field.value {
                exif::Value::Short(vals) => return vals.first().map(|&v| v as u32),
                exif::Value::Long(vals) => return vals.first().copied(),
                _ => {
                    log!("[DR] Unexpected value type for tag 0x1403");
                }
            }
        }
    }

    log!("[DR] Tag 0x1403 not found among kamadak-exif fields");
    None
}

/// Manual fallback: parse Fuji makernote IFD directly from JPEG/TIFF bytes.
///
/// Fuji makernotes start with "FUJIFILM" (8 bytes) + LE u32 offset to IFD.
/// The IFD uses standard TIFF IFD format with little-endian byte order.
fn parse_fuji_makernote_manual(data: &[u8]) -> Option<u32> {
    // Find APP1 EXIF segment in JPEG
    let exif_start = find_exif_segment(data)?;
    let tiff_base = exif_start; // TIFF header starts here
    log!("[DR] Manual: EXIF TIFF at offset {tiff_base}");

    // Parse TIFF header to get endianness and first IFD offset
    let (le, ifd0_off) = parse_tiff_header(&data[tiff_base..])?;
    log!("[DR] Manual: TIFF {} IFD0 at +{ifd0_off}", if le { "LE" } else { "BE" });

    // Walk IFD0 to find ExifIFDPointer (0x8769)
    let exif_ifd_off = find_tag_offset(data, tiff_base, ifd0_off, 0x8769, le)?;
    log!("[DR] Manual: EXIF IFD at +{exif_ifd_off}");

    // Walk EXIF IFD to find MakerNote (0x927C)
    let (mn_offset, mn_len) = find_tag_data(data, tiff_base, exif_ifd_off, 0x927C, le)?;
    log!("[DR] Manual: MakerNote at offset {mn_offset}, len {mn_len}");

    // Parse Fuji makernote: "FUJIFILM" + 4-byte LE offset to IFD
    let mn = &data[mn_offset..mn_offset + mn_len.min(data.len() - mn_offset)];
    if mn.len() < 12 || &mn[0..8] != b"FUJIFILM" {
        log!("[DR] Manual: MakerNote doesn't start with FUJIFILM");
        return None;
    }
    let mn_ifd_off = u32::from_le_bytes(mn[8..12].try_into().ok()?) as usize;
    log!("[DR] Manual: Fuji MN IFD at +{mn_ifd_off} within makernote");
    if mn_ifd_off >= mn.len() {
        log!("[DR] Manual: MN IFD offset beyond makernote data");
        return None;
    }

    // Walk the makernote IFD (always little-endian) looking for tag 0x1403
    let ifd_data = &mn[mn_ifd_off..];
    if ifd_data.len() < 2 {
        return None;
    }
    let count = u16::from_le_bytes(ifd_data[0..2].try_into().ok()?) as usize;
    log!("[DR] Manual: MN IFD has {count} entries");
    for i in 0..count {
        let entry_off = 2 + i * 12;
        if entry_off + 12 > ifd_data.len() {
            break;
        }
        let tag = u16::from_le_bytes(ifd_data[entry_off..entry_off + 2].try_into().ok()?);
        if tag == 0x1403 {
            let typ = u16::from_le_bytes(ifd_data[entry_off + 2..entry_off + 4].try_into().ok()?);
            // Type 3 = SHORT (u16), Type 4 = LONG (u32)
            let val = match typ {
                3 => u16::from_le_bytes(
                    ifd_data[entry_off + 8..entry_off + 10].try_into().ok()?,
                ) as u32,
                4 => u32::from_le_bytes(
                    ifd_data[entry_off + 8..entry_off + 12].try_into().ok()?,
                ),
                _ => {
                    log!("[DR] Manual: Tag 0x1403 has unexpected type {typ}");
                    return None;
                }
            };
            log!("[DR] Manual: Found tag 0x1403 = {val}");
            return Some(val);
        }
    }

    log!("[DR] Manual: Tag 0x1403 not found in {count} MN entries");
    None
}

/// Find the start of EXIF TIFF data inside a JPEG (after "Exif\0\0" in APP1)
fn find_exif_segment(data: &[u8]) -> Option<usize> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None; // Not JPEG
    }
    let mut pos = 2;
    while pos + 4 < data.len() {
        if data[pos] != 0xFF {
            return None;
        }
        let marker = data[pos + 1];
        let seg_len = u16::from_be_bytes(data[pos + 2..pos + 4].try_into().ok()?) as usize;
        if marker == 0xE1 {
            // APP1 — check for Exif header
            let seg_start = pos + 4;
            if seg_start + 6 < data.len() && &data[seg_start..seg_start + 6] == b"Exif\0\0" {
                return Some(seg_start + 6); // TIFF header starts here
            }
        }
        pos += 2 + seg_len;
    }
    None
}

fn parse_tiff_header(data: &[u8]) -> Option<(bool, usize)> {
    if data.len() < 8 {
        return None;
    }
    let le = match &data[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let ifd_off = if le {
        u32::from_le_bytes(data[4..8].try_into().ok()?)
    } else {
        u32::from_be_bytes(data[4..8].try_into().ok()?)
    } as usize;
    Some((le, ifd_off))
}

fn read_u16(data: &[u8], off: usize, le: bool) -> Option<u16> {
    let bytes: [u8; 2] = data.get(off..off + 2)?.try_into().ok()?;
    Some(if le { u16::from_le_bytes(bytes) } else { u16::from_be_bytes(bytes) })
}

fn read_u32(data: &[u8], off: usize, le: bool) -> Option<u32> {
    let bytes: [u8; 4] = data.get(off..off + 4)?.try_into().ok()?;
    Some(if le { u32::from_le_bytes(bytes) } else { u32::from_be_bytes(bytes) })
}

/// Walk a TIFF IFD and find the u32 value of a given tag (for pointer tags like ExifIFDPointer)
fn find_tag_offset(data: &[u8], base: usize, ifd_off: usize, target_tag: u16, le: bool) -> Option<usize> {
    let abs = base + ifd_off;
    let count = read_u16(data, abs, le)? as usize;
    for i in 0..count {
        let entry = abs + 2 + i * 12;
        let tag = read_u16(data, entry, le)?;
        if tag == target_tag {
            return Some(read_u32(data, entry + 8, le)? as usize);
        }
    }
    None
}

/// Walk a TIFF IFD and find the raw data location of a given tag
fn find_tag_data(data: &[u8], base: usize, ifd_off: usize, target_tag: u16, le: bool) -> Option<(usize, usize)> {
    let abs = base + ifd_off;
    let count = read_u16(data, abs, le)? as usize;
    for i in 0..count {
        let entry = abs + 2 + i * 12;
        let tag = read_u16(data, entry, le)?;
        if tag == target_tag {
            let typ = read_u16(data, entry + 2, le)?;
            let cnt = read_u32(data, entry + 4, le)? as usize;
            let type_size = match typ {
                1 | 2 | 7 => 1, // BYTE, ASCII, UNDEFINED
                3 => 2,         // SHORT
                4 => 4,         // LONG
                _ => 1,
            };
            let total = cnt * type_size;
            let data_off = if total <= 4 {
                entry + 8 // inline
            } else {
                base + read_u32(data, entry + 8, le)? as usize
            };
            return Some((data_off, total));
        }
    }
    None
}
