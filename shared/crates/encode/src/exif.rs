/// Build a minimal EXIF APP1 segment containing only the Orientation tag.
/// Returns the full APP1 bytes (FF E1 + length + payload) ready for insertion after SOI.
pub fn build_orientation_app1(orientation: &str) -> Vec<u8> {
    let exif_val = match orientation {
        "Rotate90" => 6u16,  // CW 90°
        "Rotate180" => 3u16,
        "Rotate270" => 8u16, // CW 270°
        _ => return Vec::new(),  // Normal — no EXIF needed
    };

    // Layout: APP1 marker(2) + length(2) + "Exif\0\0"(6) + TIFF header(8) + IFD0(18) = 36 bytes
    let tiff_data_len: u16 = 8 + 2 + 12 + 4; // header + count + 1 entry + next IFD
    let app1_length: u16 = 2 + 6 + tiff_data_len; // length field + exif header + tiff

    let mut buf = Vec::with_capacity(2 + app1_length as usize);

    // APP1 marker
    buf.extend_from_slice(&[0xFF, 0xE1]);
    buf.extend_from_slice(&app1_length.to_be_bytes());

    // EXIF header
    buf.extend_from_slice(b"Exif\0\0");

    // TIFF header (little-endian)
    buf.extend_from_slice(&[0x49, 0x49]); // "II"
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes()); // offset to IFD0

    // IFD0: 1 entry
    buf.extend_from_slice(&1u16.to_le_bytes());

    // Orientation tag (0x0112), type SHORT (3), count 1
    buf.extend_from_slice(&0x0112u16.to_le_bytes());
    buf.extend_from_slice(&3u16.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&exif_val.to_le_bytes());
    buf.extend_from_slice(&[0, 0]); // pad to 4 bytes

    // Next IFD offset = 0
    buf.extend_from_slice(&0u32.to_le_bytes());

    buf
}
