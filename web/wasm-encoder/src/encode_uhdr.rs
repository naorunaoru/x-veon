use crate::encode_jpeg;

const GAIN_MAP_QUALITY: u8 = 85;

pub fn encode(
    sdr_rgb8: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    gain_luma8: &[u8],
    gain_min: f32,
    gain_max: f32,
    offset: f32,
    hdr_capacity_max: f32,
) -> Result<Vec<u8>, String> {
    // Encode both JPEGs
    let primary_jpeg = encode_jpeg::encode(sdr_rgb8, width, height, quality)?;
    let gainmap_jpeg_raw =
        encode_jpeg::encode_grayscale(gain_luma8, width, height, GAIN_MAP_QUALITY)?;

    // Build gain map XMP (hdrgm attributes) and inject into gain map JPEG
    let gm_xmp = build_gainmap_xmp(gain_min, gain_max, offset, hdr_capacity_max);
    let gm_xmp_app1 = build_xmp_app1(&gm_xmp);
    // Gain map JPEG with XMP: SOI + XMP_APP1 + rest_of_gainmap
    let gainmap_jpeg_len = 2 + gm_xmp_app1.len() + (gainmap_jpeg_raw.len() - 2);
    let mut gainmap_jpeg = Vec::with_capacity(gainmap_jpeg_len);
    gainmap_jpeg.extend_from_slice(&[0xFF, 0xD8]); // SOI
    gainmap_jpeg.extend_from_slice(&gm_xmp_app1);
    gainmap_jpeg.extend_from_slice(&gainmap_jpeg_raw[2..]); // body after SOI

    // Build primary XMP APP1 (Container directory only, needs gainmap size)
    let primary_xmp = build_primary_xmp(gainmap_jpeg.len());
    let primary_xmp_app1 = build_xmp_app1(&primary_xmp);

    // Compute sizes for MPF offset calculation.
    // Final layout: SOI(2) + XMP_APP1 + MPF_APP2 + primary_body + gainmap_jpeg
    let primary_body = &primary_jpeg[2..]; // everything after SOI
    let mpf_data_len = mpf_payload_len();
    let mpf_app2_total = 2 + 2 + 4 + mpf_data_len; // marker(2) + length(2) + "MPF\0"(4) + data

    let primary_reconstructed_len =
        2 + primary_xmp_app1.len() + mpf_app2_total + primary_body.len();

    // MPF TIFF header starts right after "MPF\0":
    //   SOI(2) + XMP_APP1(len) + APP2_marker(2) + APP2_length(2) + "MPF\0"(4)
    let mpf_tiff_offset = 2 + primary_xmp_app1.len() + 2 + 2 + 4;
    let image2_offset = primary_reconstructed_len - mpf_tiff_offset;

    let mpf_app2 = build_mpf(
        primary_reconstructed_len as u32,
        gainmap_jpeg.len() as u32,
        image2_offset as u32,
    );

    // Assemble final file
    let mut out = Vec::with_capacity(primary_reconstructed_len + gainmap_jpeg.len());
    out.extend_from_slice(&[0xFF, 0xD8]); // SOI
    out.extend_from_slice(&primary_xmp_app1);
    out.extend_from_slice(&mpf_app2);
    out.extend_from_slice(primary_body);
    out.extend_from_slice(&gainmap_jpeg);

    Ok(out)
}

/// Primary image XMP: Container directory + hdrgm:Version only.
fn build_primary_xmp(gainmap_size: usize) -> String {
    let body = format!(
        r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"
      xmlns:Container="http://ns.google.com/photos/1.0/container/"
      xmlns:Item="http://ns.google.com/photos/1.0/container/item/"
      hdrgm:Version="1.0">
      <Container:Directory>
        <rdf:Seq>
          <rdf:li rdf:parseType="Resource">
            <Container:Item
              Item:Semantic="Primary"
              Item:Mime="image/jpeg"/>
          </rdf:li>
          <rdf:li rdf:parseType="Resource">
            <Container:Item
              Item:Semantic="GainMap"
              Item:Mime="image/jpeg"
              Item:Length="{}"/>
          </rdf:li>
        </rdf:Seq>
      </Container:Directory>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"#,
        gainmap_size
    );
    format!("<?xpacket begin='\u{FEFF}' id='W5M0MpCehiHzreSzNTczkc9d'?>\n{body}\n<?xpacket end='w'?>")
}

/// Gain map image XMP: all hdrgm numeric attributes.
fn build_gainmap_xmp(gain_min: f32, gain_max: f32, offset: f32, hdr_capacity_max: f32) -> String {
    let body = format!(
        r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"
      hdrgm:Version="1.0"
      hdrgm:GainMapMin="{:.6}"
      hdrgm:GainMapMax="{:.6}"
      hdrgm:Gamma="1.0"
      hdrgm:OffsetSDR="{:.6}"
      hdrgm:OffsetHDR="{:.6}"
      hdrgm:HDRCapacityMin="0.0"
      hdrgm:HDRCapacityMax="{:.6}"
      hdrgm:BaseRenditionIsHDR="False"/>
  </rdf:RDF>
</x:xmpmeta>"#,
        gain_min, gain_max, offset, offset, hdr_capacity_max
    );
    format!("<?xpacket begin='\u{FEFF}' id='W5M0MpCehiHzreSzNTczkc9d'?>\n{body}\n<?xpacket end='w'?>")
}

fn build_xmp_app1(xmp: &str) -> Vec<u8> {
    let xmp_bytes = xmp.as_bytes();
    let ns = b"http://ns.adobe.com/xap/1.0/\0";
    let payload_len = ns.len() + xmp_bytes.len();
    // JPEG marker length field includes itself (2 bytes) but not the marker bytes
    let marker_len = (2 + payload_len) as u16;

    let mut app1 = Vec::with_capacity(2 + marker_len as usize);
    app1.extend_from_slice(&[0xFF, 0xE1]); // APP1 marker
    app1.extend_from_slice(&marker_len.to_be_bytes()); // big-endian length (JPEG convention)
    app1.extend_from_slice(ns);
    app1.extend_from_slice(xmp_bytes);
    app1
}

/// Size of the MPF TIFF payload (after "MPF\0" identifier).
fn mpf_payload_len() -> usize {
    // TIFF header(8) + IFD count(2) + 3 entries(36) + next IFD(4) + MP Entry data(32)
    8 + 2 + 3 * 12 + 4 + 32
}

fn build_mpf(primary_size: u32, gainmap_size: u32, image2_offset: u32) -> Vec<u8> {
    let data_len = mpf_payload_len();
    // APP2 length field = 2 (self) + 4 ("MPF\0") + data
    let marker_len = (2 + 4 + data_len) as u16;

    let mut buf = Vec::with_capacity(2 + marker_len as usize);

    // APP2 marker
    buf.extend_from_slice(&[0xFF, 0xE2]);
    buf.extend_from_slice(&marker_len.to_be_bytes()); // big-endian (JPEG convention)

    // MPF identifier
    buf.extend_from_slice(b"MPF\0");

    // --- TIFF structure (little-endian from here) ---

    // TIFF header
    buf.extend_from_slice(&[0x49, 0x49]); // "II" = little-endian
    buf.extend_from_slice(&42u16.to_le_bytes()); // TIFF magic
    buf.extend_from_slice(&8u32.to_le_bytes()); // offset to first IFD

    // IFD: 3 entries
    buf.extend_from_slice(&3u16.to_le_bytes());

    // Entry 1: MPFVersion (tag=0xB000, type=UNDEFINED=7, count=4, value="0100")
    buf.extend_from_slice(&0xB000u16.to_le_bytes());
    buf.extend_from_slice(&7u16.to_le_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(b"0100");

    // Entry 2: NumberOfImages (tag=0xB001, type=LONG=4, count=1, value=2)
    buf.extend_from_slice(&0xB001u16.to_le_bytes());
    buf.extend_from_slice(&4u16.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&2u32.to_le_bytes());

    // Entry 3: MPEntry (tag=0xB002, type=UNDEFINED=7, count=32, offset to data)
    // MP Entry data starts after IFD: offset 8 (header) + 2 (count) + 36 (entries) + 4 (next) = 50
    let mp_entry_offset: u32 = 8 + 2 + 3 * 12 + 4;
    buf.extend_from_slice(&0xB002u16.to_le_bytes());
    buf.extend_from_slice(&7u16.to_le_bytes());
    buf.extend_from_slice(&32u32.to_le_bytes());
    buf.extend_from_slice(&mp_entry_offset.to_le_bytes());

    // Next IFD offset = 0
    buf.extend_from_slice(&0u32.to_le_bytes());

    // MP Entry 1 (primary): type=primary (matches libultrahdr convention)
    buf.extend_from_slice(&0x030000u32.to_le_bytes()); // attributes
    buf.extend_from_slice(&primary_size.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // offset 0 for primary
    buf.extend_from_slice(&0u16.to_le_bytes()); // dependent 1
    buf.extend_from_slice(&0u16.to_le_bytes()); // dependent 2

    // MP Entry 2 (gain map)
    buf.extend_from_slice(&0u32.to_le_bytes()); // attributes
    buf.extend_from_slice(&gainmap_size.to_le_bytes());
    buf.extend_from_slice(&image2_offset.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // dependent 1
    buf.extend_from_slice(&0u16.to_le_bytes()); // dependent 2

    buf
}
