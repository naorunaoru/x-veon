"""Examine RAF files: find compressed raw data and parse Fuji compressed headers."""
import sys
import struct
import rawpy

TAG_RAF_OFFSETS = 0xF00D

def parse_tiff_ifd(data, offset, little_endian=True):
    """Parse a TIFF IFD returning list of (tag, type, count, value_or_offset)."""
    fmt = '<' if little_endian else '>'
    if offset + 2 > len(data):
        return [], 0
    num_entries = struct.unpack_from(f'{fmt}H', data, offset)[0]
    entries = []
    for i in range(num_entries):
        entry_off = offset + 2 + i * 12
        if entry_off + 12 > len(data):
            break
        tag, dtype, count = struct.unpack_from(f'{fmt}HHI', data, entry_off)
        val = struct.unpack_from(f'{fmt}I', data, entry_off + 8)[0]
        entries.append((tag, dtype, count, val))
    # Next IFD pointer
    next_ptr_off = offset + 2 + num_entries * 12
    next_ifd = 0
    if next_ptr_off + 4 <= len(data):
        next_ifd = struct.unpack_from(f'{fmt}I', data, next_ptr_off)[0]
    return entries, next_ifd


def examine_raf(path):
    print(f"\n{'='*60}")
    print(f"File: {path}")

    with rawpy.imread(path) as raw:
        h, w = raw.raw_image.shape
        print(f"  Sensor: {w}x{h}")
        expected_bytes = w * h * 2  # 16bpp

    with open(path, 'rb') as f:
        data = f.read()

    camera = data[0x1C:0x3C].rstrip(b'\x00').decode('ascii', errors='replace')
    print(f"  Camera: {camera}")

    cfa_data_off = struct.unpack('>I', data[0x64:0x68])[0]
    cfa_data_len = struct.unpack('>I', data[0x68:0x6C])[0]
    print(f"  CFA data: offset=0x{cfa_data_off:x}, len={cfa_data_len:,}")
    print(f"  Compressed: {'YES' if cfa_data_len < expected_bytes else 'NO'}")

    # The CFA data section is a TIFF
    tiff_start = cfa_data_off
    le = data[tiff_start:tiff_start+2] == b'II'
    fmt = '<' if le else '>'
    ifd0_off = struct.unpack_from(f'{fmt}I', data, tiff_start + 4)[0]

    # Walk all IFDs
    ifd_abs = tiff_start + ifd0_off
    ifd_num = 0
    while ifd_abs > 0 and ifd_abs < len(data) - 2:
        entries, next_ifd = parse_tiff_ifd(data, ifd_abs, le)
        ifd_num += 1
        print(f"\n  IFD#{ifd_num} at 0x{ifd_abs:x} ({len(entries)} tags):")
        for tag, dtype, count, val in entries:
            print(f"    tag=0x{tag:04x} type={dtype} count={count} val={val} (0x{val:x})")
            if tag == TAG_RAF_OFFSETS:
                # rawloader: offset = tag_value + ifd.start_offset
                # In rawloader, a TIFF IFD's start_offset is the position
                # of the IFD data origin. For sub-IFDs in RAF, this is
                # the position of the IFD itself... let me try:
                for base_name, base in [("tiff_start", tiff_start), ("ifd_abs", ifd_abs)]:
                    candidate = val + base
                    if candidate + 2 <= len(data):
                        sig = struct.unpack('>H', data[candidate:candidate+2])[0]
                        print(f"      → 0x{candidate:x} ({base_name}+val) sig=0x{sig:04x} {'✓ FUJI HEADER' if sig == 0x4953 else ''}")

        if next_ifd == 0:
            break
        ifd_abs = tiff_start + next_ifd

    # Also search for 0x4953 signature in a wider range
    print(f"\n  Searching for Fuji compressed header (0x4953)...")
    search_start = cfa_data_off
    search_end = min(len(data) - 16, cfa_data_off + cfa_data_len)
    found = False
    for off in range(search_start, min(search_start + 100000, search_end)):
        if data[off] == 0x49 and data[off+1] == 0x53:
            sig = struct.unpack('>H', data[off:off+2])[0]
            if sig == 0x4953:
                # Validate: check version and raw_type
                ver = data[off+2]
                rt = data[off+3]
                if ver == 1 and rt in (0, 16):
                    print(f"  Found at 0x{off:x}!")
                    parse_fuji_header(data, off)
                    found = True
                    break
    if not found:
        print(f"  Not found — likely uncompressed")


def parse_fuji_header(data, off):
    signature = struct.unpack('>H', data[off:off+2])[0]
    version = data[off+2]
    raw_type = data[off+3]
    raw_bits = data[off+4]
    raw_height = struct.unpack('>H', data[off+5:off+7])[0]
    raw_rounded_width = struct.unpack('>H', data[off+7:off+9])[0]
    raw_width = struct.unpack('>H', data[off+9:off+11])[0]
    block_size = struct.unpack('>H', data[off+11:off+13])[0]
    blocks_in_row = data[off+13]
    total_lines = struct.unpack('>H', data[off+14:off+16])[0]

    print(f"  Fuji Compressed Header:")
    print(f"    signature:   0x{signature:04x}")
    print(f"    version:     {version}")
    print(f"    raw_type:    {raw_type} ({'X-Trans' if raw_type == 16 else 'Bayer'})")
    print(f"    raw_bits:    {raw_bits}")
    print(f"    raw_height:  {raw_height}")
    print(f"    raw_rnd_w:   {raw_rounded_width}")
    print(f"    raw_width:   {raw_width}")
    print(f"    block_size:  {block_size} (0x{block_size:x})")
    print(f"    blocks_row:  {blocks_in_row}")
    print(f"    total_lines: {total_lines}")
    print(f"    line_width:  {block_size * 2 // 3}")

    # Read block sizes
    bs_off = off + 16
    block_sizes = []
    for i in range(blocks_in_row):
        bs = struct.unpack('>I', data[bs_off + i*4:bs_off + i*4 + 4])[0]
        block_sizes.append(bs)
    print(f"    block_sizes: {block_sizes}")
    print(f"    total strip: {sum(block_sizes):,}")


if __name__ == '__main__':
    for f in sys.argv[1:]:
        examine_raf(f)
