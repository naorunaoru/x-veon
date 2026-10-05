use xveon_encode::{encode, EncodeOptions, Format};
const ONE: EncodeOptions = EncodeOptions { threads: 1 };
fn image(w: u32, h: u32, scale: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h { for x in 0..w { for c in 0..3 {
        v.push(scale * ((x * 7 + y * 13 + c * 29) % 97) as f32 / 96.0);
    } } }
    v
}
#[test] fn rejects_a_length_mismatch() {
    let err = encode(&[0.0; 5], &[], 2, 1, "Normal", Format::Tiff, 95, 100.0, ONE).unwrap_err();
    assert!(err.contains("data length mismatch"), "{err}");
}
#[test] fn rejects_an_unknown_format() { assert_eq!(Format::parse("png").unwrap_err(), "unknown format: png"); }
#[test] fn rejects_empty_images() { assert!(encode(&[], &[], 0, 0, "Normal", Format::Tiff, 95, 100.0, ONE).is_err()); }
#[test] fn tiff_is_rotated_and_sized() {
    let px = image(3, 2, 1.0);
    let normal = encode(&px, &[], 3, 2, "Normal", Format::Tiff, 95, 100.0, ONE).unwrap();
    assert_eq!(&normal[..4], b"II*\0");
    let rotated = encode(&px, &[], 3, 2, "Rotate90", Format::Tiff, 95, 100.0, ONE).unwrap();
    let mut d = tiff::decoder::Decoder::new(std::io::Cursor::new(rotated)).unwrap();
    assert_eq!(d.dimensions().unwrap(), (2, 3));
}
#[test] fn avif_has_a_container_and_is_deterministic() {
    let px = image(64, 48, 4.0);
    let a = encode(&px, &[], 64, 48, "Normal", Format::Avif, 95, 1000.0, ONE).unwrap();
    let b = encode(&px, &[], 64, 48, "Normal", Format::Avif, 95, 1000.0, ONE).unwrap();
    assert_eq!(&a[4..8], b"ftyp"); assert!(a.windows(4).any(|w| w == b"avif")); assert_eq!(a, b);
}
#[test] fn ultra_hdr_has_two_jpegs_and_mpf() {
    let (sdr, hdr) = (image(64, 48, 1.0), image(64, 48, 6.0));
    let j = encode(&sdr, &hdr, 64, 48, "Rotate90", Format::JpegHdr, 95, 1000.0, ONE).unwrap();
    assert_eq!(&j[..2], &[0xFF, 0xD8]);
    assert!(j.windows(4).any(|w| w == b"MPF\0"));
    assert!(j.windows(13).any(|w| w == b"hdrgm:Version"));
    assert!(j.windows(3).filter(|w| *w == [0xFF, 0xD8, 0xFF]).count() >= 2);
}
#[test] fn ultra_hdr_rejects_jpeg_sized_overflow() {
    assert!(encode(&vec![0.0; 70_000 * 3], &vec![0.0; 70_000 * 3], 70_000, 1, "Normal", Format::JpegHdr, 95, 1000.0, ONE).is_err());
}

#[test] fn threads_do_not_change_any_format() {
    let (sdr, hdr) = (image(700, 500, 1.0), image(700, 500, 6.0));
    let four = EncodeOptions { threads: 4 };
    let a = |t| encode(&hdr, &[], 700, 500, "Normal", Format::Avif, 90, 1000.0, t).unwrap();
    let j = |t| encode(&sdr, &hdr, 700, 500, "Rotate90", Format::JpegHdr, 90, 1000.0, t).unwrap();
    let f = |t| encode(&sdr, &[], 700, 500, "Rotate180", Format::Tiff, 90, 100.0, t).unwrap();
    assert_eq!(a(ONE), a(four));
    assert_eq!(j(ONE), j(four));
    assert_eq!(f(ONE), f(four));
}
