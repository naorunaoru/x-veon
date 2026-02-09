"""
Fuji compressed RAF decompressor — Python reference implementation.
Based on RawSpeed's FujiDecompressor.cpp by:
  Alexey Danilchenko, Alex Tutubalin, Uwe Müssel, Roman Lebedev
"""
import struct
import numpy as np

# ---------------------------------------------------------------------------
# Line buffer row indices (18 total = 5R + 8G + 5B)
# ---------------------------------------------------------------------------
R0, R1, R2, R3, R4 = 0, 1, 2, 3, 4
G0, G1, G2, G3, G4, G5, G6, G7 = 5, 6, 7, 8, 9, 10, 11, 12
B0, B1, B2, B3, B4 = 13, 14, 15, 16, 17
LTOTAL = 18

# Per-row color assignments (RGGB CFA applied to 6 rows)
# Row 0: R→R2, G→G2 | Row 1: G→G3, B→B2 | Row 2: R→R3, G→G4
# Row 3: G→G5, B→B3 | Row 4: R→R4, G→G6 | Row 5: G→G7, B→B4
ROW_LINES = [
    (R2, G2), (G3, B2), (R3, G4),
    (G5, B3), (R4, G6), (G7, B4),
]

# Which extend function to call after each row: 0=R, 1=G, 2=B
ROW_EXTENDS = [
    (0, 1), (1, 2), (0, 1),
    (1, 2), (0, 1), (1, 2),
]

# X-Trans interpolation pattern: True = interpolation (no bitstream), False = sample
# Indexed by (row, comp, i%2) where applicable
def is_interpolation(row, comp, i):
    """Whether this (row, comp, i) position uses interpolation vs sample decoding."""
    if comp == 0:
        if row == 0 or row == 5:
            return True
        if row == 2 and i % 2 == 0:
            return True
        if row == 4 and i % 2 != 0:
            return True
        return False
    else:  # comp == 1
        if row == 1 or row == 2:
            return True
        if row == 3 and i % 2 != 0:
            return True
        if row == 5 and i % 2 == 0:
            return True
        return False


# ---------------------------------------------------------------------------
# Bitstream reader (MSB-first)
# ---------------------------------------------------------------------------
class BitPumpMSB:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0       # byte position
        self.bits = 0      # bit buffer
        self.nbits = 0     # number of valid bits in buffer

    def _fill(self):
        """Load 32 more bits from data."""
        if self.pos + 4 <= len(self.data):
            val = struct.unpack_from('>I', self.data, self.pos)[0]
        else:
            # Near end of data, read carefully
            val = 0
            for i in range(4):
                if self.pos + i < len(self.data):
                    val = (val << 8) | self.data[self.pos + i]
                else:
                    val <<= 8
        self.bits = (self.bits << 32) | val
        self.pos += 4
        self.nbits += 32

    def peek(self, n):
        while n > self.nbits:
            self._fill()
        return (self.bits >> (self.nbits - n)) & ((1 << n) - 1)

    def skip(self, n):
        self.nbits -= n
        self.bits &= (1 << self.nbits) - 1

    def get(self, n):
        if n == 0:
            return 0
        val = self.peek(n)
        self.skip(n)
        return val


# ---------------------------------------------------------------------------
# Header and params
# ---------------------------------------------------------------------------
class FujiHeader:
    def __init__(self, data, offset):
        self.signature = struct.unpack_from('>H', data, offset)[0]
        self.version = data[offset + 2]
        self.raw_type = data[offset + 3]
        self.raw_bits = data[offset + 4]
        self.raw_height = struct.unpack_from('>H', data, offset + 5)[0]
        self.raw_rounded_width = struct.unpack_from('>H', data, offset + 7)[0]
        self.raw_width = struct.unpack_from('>H', data, offset + 9)[0]
        self.block_size = struct.unpack_from('>H', data, offset + 11)[0]
        self.blocks_in_row = data[offset + 13]
        self.total_lines = struct.unpack_from('>H', data, offset + 14)[0]

        assert self.signature == 0x4953, f"Bad signature: {self.signature:#x}"
        assert self.version == 1
        assert self.raw_type == 16, "Only X-Trans supported"

    def __repr__(self):
        return (f"FujiHeader(bits={self.raw_bits}, {self.raw_width}x{self.raw_height}, "
                f"block={self.block_size}, strips={self.blocks_in_row}, lines={self.total_lines})")


class FujiParams:
    def __init__(self, header: FujiHeader):
        h = header
        assert h.block_size % 3 == 0
        self.line_width = (h.block_size * 2) // 3

        self.q_point = [0, 0x12, 0x43, 0x114, (1 << h.raw_bits) - 1]
        self.min_value = 0x40

        # Gradient lookup table
        n = 2 * (1 << h.raw_bits)
        self.q_table = [0] * n
        for i in range(n):
            self.q_table[i] = self._get_gradient(i)

        if self.q_point[4] == 0x3FFF:  # 14-bit
            self.total_values = 0x4000
            self.raw_bits = 14
            self.max_bits = 56
            self.max_diff = 256
        elif self.q_point[4] == 0xFFFF:  # 16-bit
            self.total_values = 0x10000
            self.raw_bits = 16
            self.max_bits = 64
            self.max_diff = 1024
        else:
            raise ValueError(f"Unsupported q_point[4] = {self.q_point[4]}")

    def _get_gradient(self, cur_val):
        v = cur_val - self.q_point[4]
        av = abs(v)
        grad = 0
        if av > 0: grad = 1
        if av >= self.q_point[1]: grad = 2
        if av >= self.q_point[2]: grad = 3
        if av >= self.q_point[3]: grad = 4
        return -grad if v < 0 else grad

    def quant_gradient(self, v1, v2):
        return 9 * self.q_table[self.q_point[4] + v1] + self.q_table[self.q_point[4] + v2]


# ---------------------------------------------------------------------------
# Core decode functions
# ---------------------------------------------------------------------------
def fuji_zerobits(pump: BitPumpMSB) -> int:
    count = 0
    while True:
        batch = pump.peek(32)
        # Count leading zeros
        if batch == 0:
            zeros = 32
        else:
            zeros = 0
            tmp = batch
            while (tmp & 0x80000000) == 0:
                zeros += 1
                tmp <<= 1
        count += zeros
        if zeros < 32:
            pump.skip(zeros + 1)  # skip zeros + the terminating 1
            break
        pump.skip(32)
    return count


def bit_diff(value1, value2):
    """How many times must value2 be doubled to be >= value1?"""
    if value1 <= 0:
        return 0
    if value2 <= 0:
        return 15
    lz1 = (value1).bit_length()
    lz2 = (value2).bit_length()
    # leading zeros in 32-bit: 32 - bit_length
    lz1_clz = 32 - lz1
    lz2_clz = 32 - lz2
    dec_bits = max(lz2_clz - lz1_clz, 0)
    if (value2 << dec_bits) < value1:
        dec_bits += 1
    return min(dec_bits, 15)


def fuji_decode_sample(pump, params, grad, interp_val, grads):
    """Decode one sample using adaptive entropy coding."""
    gradient = abs(grad)

    pump_pos_before = pump.pos
    sample_bits = fuji_zerobits(pump)

    if sample_bits < params.max_bits - params.raw_bits - 1:
        code_bits = bit_diff(grads[gradient][0], grads[gradient][1])
        code_delta = sample_bits << code_bits
    else:
        code_bits = params.raw_bits
        code_delta = 1

    code = pump.get(code_bits) if code_bits else 0
    code += code_delta

    if code < 0 or code >= params.total_values:
        raise ValueError(
            f"fuji_decode_sample (sample #{_sample_count}): code={code} out of range "
            f"(sample_bits={sample_bits}, code_bits={code_bits}, code_delta={code_delta}, "
            f"gradient={gradient}, grads={grads[gradient]}, "
            f"pump_pos_before={pump_pos_before})"
        )

    # Signed zigzag decoding
    if code & 1:
        code = -1 - code // 2
    else:
        code = code // 2

    # Update gradient statistics
    grads[gradient][0] += abs(code)
    if grads[gradient][1] == params.min_value:
        grads[gradient][0] >>= 1
        grads[gradient][1] >>= 1
    grads[gradient][1] += 1

    # Apply code to interpolation value
    if grad < 0:
        interp_val -= code
    else:
        interp_val += code

    # Wrap around
    if interp_val < 0:
        interp_val += params.total_values
    elif interp_val > params.q_point[4]:
        interp_val -= params.total_values

    return max(0, min(interp_val, params.q_point[4]))


# ---------------------------------------------------------------------------
# Interpolation (prediction from neighbors)
# ---------------------------------------------------------------------------
def L(lines, stride, row, col):
    """Access line buffer."""
    return int(lines[row * stride + col])


def interpolation_even_inner(lines, stride, params, c, col):
    """Predict even pixel from neighbors, return (grad, interp_val)."""
    Rb = L(lines, stride, c - 1, 1 + 2 * col)
    Rc = L(lines, stride, c - 1, 2 * col)              # 1 + 2*(col-1) + 1
    Rd = L(lines, stride, c - 1, 2 * col + 2)          # 1 + 2*col + 1
    Rf = L(lines, stride, c - 2, 1 + 2 * col)

    dRcRb = abs(Rc - Rb)
    dRfRb = abs(Rf - Rb)
    dRdRb = abs(Rd - Rb)

    term0 = 2 * Rb
    if dRcRb > max(dRfRb, dRdRb):
        term1, term2 = Rf, Rd
    else:
        term1 = Rf if dRdRb > max(dRcRb, dRfRb) else Rd
        term2 = Rc

    interp_val = (term0 + term1 + term2) >> 2
    grad = params.quant_gradient(Rb - Rf, Rc - Rb)
    return grad, interp_val


def interpolation_odd_inner(lines, stride, params, c, col):
    """Predict odd pixel from neighbors, return (grad, interp_val)."""
    Ra = L(lines, stride, c, 1 + 2 * col)               # same row, even
    Rb = L(lines, stride, c - 1, 1 + 2 * col + 1)       # prev row, odd
    Rc = L(lines, stride, c - 1, 1 + 2 * col)            # prev row, even
    Rd = L(lines, stride, c - 1, 1 + 2 * (col + 1))     # prev row, next col, even
    Rg = L(lines, stride, c, 1 + 2 * (col + 1))          # same row, next col, even

    interp_val = Ra + Rg
    mn, mx = min(Rc, Rd), max(Rc, Rd)
    if Rb < mn or Rb > mx:
        interp_val += 2 * Rb
        interp_val >>= 1
    interp_val >>= 1

    grad = params.quant_gradient(Rb - Rc, Rc - Ra)
    return grad, interp_val


# ---------------------------------------------------------------------------
# Block decode (6 rows of one MCU line)
# ---------------------------------------------------------------------------
def extend_generic(lines, stride, start, end):
    for i in range(start, end + 1):
        lines[i * stride + 0] = lines[(i - 1) * stride + 1]
        lines[i * stride + stride - 1] = lines[(i - 1) * stride + stride - 2]


def extend_red(lines, stride):
    extend_generic(lines, stride, R2, R4)

def extend_green(lines, stride):
    extend_generic(lines, stride, G2, G7)

def extend_blue(lines, stride):
    extend_generic(lines, stride, B2, B4)

EXTEND_FNS = [extend_red, extend_green, extend_blue]


_sample_count = 0

def xtrans_decode_block(pump, params, lines, stride, grad_even, grad_odd):
    """Decode one 6-row MCU line."""
    global _sample_count
    half_lw = params.line_width // 2

    for row in range(6):
        c0, c1 = ROW_LINES[row]
        grad_idx = row % 3

        col_even = [0, 0]  # per-component even column counter
        col_odd = [0, 0]   # per-component odd column counter

        for i in range(half_lw + 4):
            # Decode even pixels
            if i < half_lw:
                for comp in range(2):
                    c = c0 if comp == 0 else c1
                    col = col_even[comp]

                    if is_interpolation(row, comp, i):
                        # Pure interpolation, no bitstream
                        _, interp_val = interpolation_even_inner(lines, stride, params, c, col)
                        sample = interp_val
                    else:
                        # Entropy-coded sample
                        grad, interp_val = interpolation_even_inner(lines, stride, params, c, col)
                        sample = fuji_decode_sample(pump, params, grad, interp_val, grad_even[grad_idx])
                        _sample_count += 1

                    lines[c * stride + 1 + 2 * col] = sample
                    col_even[comp] += 1

            # Decode odd pixels (start 4 positions behind even)
            if i >= 4:
                for comp in range(2):
                    c = c0 if comp == 0 else c1
                    col = col_odd[comp]

                    grad, interp_val = interpolation_odd_inner(lines, stride, params, c, col)
                    sample = fuji_decode_sample(pump, params, grad, interp_val, grad_odd[grad_idx])
                    _sample_count += 1

                    lines[c * stride + 1 + 2 * col + 1] = sample
                    col_odd[comp] += 1

        # Extend helper columns
        for ext_idx in ROW_EXTENDS[row]:
            EXTEND_FNS[ext_idx](lines, stride)


# ---------------------------------------------------------------------------
# Copy decoded lines to output image
# ---------------------------------------------------------------------------
# X-Trans CFA at phase (0,0) — determines which line buffer row each pixel reads from
# 0=R, 1=G, 2=B
XTRANS_CFA = np.array([
    [1, 1, 0, 1, 1, 2],  # G G R G G B
    [1, 1, 2, 1, 1, 0],  # G G B G G R
    [2, 0, 1, 0, 2, 1],  # B R G R B G
    [1, 1, 2, 1, 1, 0],  # G G B G G R
    [1, 1, 0, 1, 1, 2],  # G G R G G B
    [0, 2, 1, 2, 0, 1],  # R B G B R G
], dtype=np.uint8)


def xtrans_col_index(img_col):
    """Map image column to line buffer column."""
    return (((img_col * 2 // 3) & 0x7FFFFFFE) | ((img_col % 3) & 1)) + ((img_col % 3) >> 1)


def copy_line_to_xtrans(lines, stride, header, strip_width, strip_offset_x, cur_line, out, out_width):
    """Copy decoded line buffer to output image using X-Trans CFA mapping."""
    num_mcus_x = strip_width // 6

    for mcu_x in range(num_mcus_x):
        for mcu_row in range(6):
            for mcu_col in range(6):
                img_row = mcu_row
                img_col = 6 * mcu_x + mcu_col

                color = XTRANS_CFA[mcu_row, mcu_col]

                if color == 0:    # RED
                    row = R2 + (img_row >> 1)
                elif color == 1:  # GREEN
                    row = G2 + img_row
                else:             # BLUE
                    row = B2 + (img_row >> 1)

                buf_col = 1 + xtrans_col_index(img_col)
                val = lines[row * stride + buf_col]

                out_y = 6 * cur_line + mcu_row
                out_x = strip_offset_x + img_col

                if 0 <= out_y < out.shape[0] and 0 <= out_x < out.shape[1]:
                    out[out_y, out_x] = val


# ---------------------------------------------------------------------------
# Top-level strip decoder
# ---------------------------------------------------------------------------
def decode_strip(strip_data, header, params, strip_width, strip_offset_x, out, out_width):
    """Decode one vertical strip."""
    stride = params.line_width + 2
    lines = np.zeros(LTOTAL * stride, dtype=np.int32)

    init_pair = [params.max_diff, 1]
    grad_even = [[list(init_pair) for _ in range(41)] for _ in range(3)]
    grad_odd = [[list(init_pair) for _ in range(41)] for _ in range(3)]

    pump = BitPumpMSB(strip_data)

    for cur_line in range(header.total_lines):
        if cur_line > 0:
            # Rotate: last 2 lines → first 2
            line_bytes = stride  # number of values per line
            for start, count in [(R0, 5), (G0, 8), (B0, 5)]:
                src_start = (start + count - 2) * stride
                dst_start = start * stride
                lines[dst_start:dst_start + 2 * line_bytes] = lines[src_start:src_start + 2 * line_bytes]

            # Set helper column for first decoded line
            for start, _ in [(R0, 5), (G0, 8), (B0, 5)]:
                row = start + 2
                prev = start + 1
                lines[row * stride + stride - 1] = lines[prev * stride + stride - 2]

        # Decode this MCU line
        xtrans_decode_block(pump, params, lines, stride, grad_even, grad_odd)

        # Copy to output
        copy_line_to_xtrans(lines, stride, header, strip_width, strip_offset_x, cur_line, out, out_width)

        if cur_line % 100 == 0:
            print(f"    strip line {cur_line}/{header.total_lines}")


# ---------------------------------------------------------------------------
# Main decoder
# ---------------------------------------------------------------------------
def decode_fuji_compressed(file_data, compressed_offset):
    """Decode a Fuji compressed RAF from file data."""
    header = FujiHeader(file_data, compressed_offset)
    params = FujiParams(header)
    print(f"Header: {header}")
    print(f"Params: line_width={params.line_width}, raw_bits={params.raw_bits}, "
          f"max_bits={params.max_bits}, max_diff={params.max_diff}")

    # Read block sizes
    bs_off = compressed_offset + 16
    block_sizes = []
    for i in range(header.blocks_in_row):
        bs = struct.unpack_from('>I', file_data, bs_off + i * 4)[0]
        block_sizes.append(bs)

    # Compute strip data start (after header + block sizes + padding)
    pos = 16 + header.blocks_in_row * 4
    if pos & 0xC:
        pos += 0x10 - (pos & 0xC)
    data_start = compressed_offset + pos

    # Compute strip offsets
    strip_offsets = []
    off = data_start
    for bs in block_sizes:
        strip_offsets.append(off)
        off += bs

    print(f"Strips: {header.blocks_in_row}, block_sizes={block_sizes}")

    # Allocate output
    out = np.zeros((header.raw_height, header.raw_width), dtype=np.uint16)

    # Decode each strip
    for block in range(header.blocks_in_row):
        strip_data = file_data[strip_offsets[block]:strip_offsets[block] + block_sizes[block]]
        strip_width = header.block_size if block + 1 < header.blocks_in_row else header.raw_width - header.block_size * block
        strip_offset_x = header.block_size * block

        print(f"  Decoding strip {block}: width={strip_width}, offset_x={strip_offset_x}, "
              f"data_size={block_sizes[block]:,}")

        decode_strip(strip_data, header, params, strip_width, strip_offset_x, out, header.raw_width)

    return out


if __name__ == '__main__':
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/4T/naoru/DCIM/107_FUJI/DSCF7001.RAF"

    with open(path, 'rb') as f:
        file_data = f.read()

    # Find compressed header
    cfa_data_off = struct.unpack('>I', file_data[0x64:0x68])[0]

    # Search for 0x4953 signature
    compressed_offset = None
    for off in range(cfa_data_off, min(len(file_data) - 16, cfa_data_off + 100000)):
        if file_data[off] == 0x49 and file_data[off + 1] == 0x53:
            if file_data[off + 2] == 1 and file_data[off + 3] in (0, 16):
                compressed_offset = off
                break

    if compressed_offset is None:
        print("Could not find Fuji compressed header!")
        sys.exit(1)

    print(f"Compressed header at 0x{compressed_offset:x}")
    result = decode_fuji_compressed(file_data, compressed_offset)

    np.save("/Users/naoru/projects/xtrans-demosaic/decoder_output.npy", result)
    print("Saved decoder_output.npy")

    # Compare with reference
    ref = np.load("/Users/naoru/projects/xtrans-demosaic/reference_full.npy")
    print(f"\nResult shape: {result.shape}, Reference shape: {ref.shape}")
    print(f"Result range: [{result.min()}, {result.max()}]")
    print(f"Reference range: [{ref.min()}, {ref.max()}]")

    # Compare a patch
    patch_ref = ref[:96, :96]
    patch_out = result[:96, :96]
    diff = np.abs(patch_ref.astype(np.int32) - patch_out.astype(np.int32))
    print(f"\n96x96 patch comparison:")
    print(f"  Max diff: {diff.max()}")
    print(f"  Mean diff: {diff.mean():.4f}")
    print(f"  Exact match: {np.array_equal(patch_ref, patch_out)}")

    if not np.array_equal(ref, result):
        # Find first mismatch
        mismatches = np.argwhere(ref != result)
        if len(mismatches) > 0:
            y, x = mismatches[0]
            print(f"\nFirst mismatch at ({y}, {x}): ref={ref[y,x]}, got={result[y,x]}")
            print(f"Total mismatches: {len(mismatches)} / {ref.size}")
    else:
        print("\nPERFECT MATCH!")
