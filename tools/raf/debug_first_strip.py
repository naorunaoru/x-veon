"""Debug: verify strip data offset and first bytes."""
import struct

RAF_PATH = "/Volumes/4T/naoru/DCIM/107_FUJI/DSCF7001.RAF"

with open(RAF_PATH, 'rb') as f:
    data = f.read()

# Find compressed header
cfa_data_off = struct.unpack('>I', data[0x64:0x68])[0]
for off in range(cfa_data_off, cfa_data_off + 100000):
    if data[off] == 0x49 and data[off+1] == 0x53 and data[off+2] == 1:
        hdr_off = off
        break

print(f"Compressed header at: 0x{hdr_off:x}")

# Parse header
blocks_in_row = data[hdr_off + 13]
print(f"blocks_in_row: {blocks_in_row}")

# Block sizes
bs_off = hdr_off + 16
block_sizes = []
for i in range(blocks_in_row):
    bs = struct.unpack_from('>I', data, bs_off + i*4)[0]
    block_sizes.append(bs)
print(f"block_sizes: {block_sizes}")

# Data offset calculation
raw_offset_check = 4 * blocks_in_row  # 32
padding = 0
if raw_offset_check & 0xC:
    padding = 0x10 - (raw_offset_check & 0xC)
print(f"raw_offset for padding check: {raw_offset_check}, padding: {padding}")

strip0_off = hdr_off + 16 + blocks_in_row * 4 + padding
print(f"Strip 0 data at: 0x{strip0_off:x}")

# Show bytes at the strip offset
print(f"\nBytes at strip 0 start (hex):")
for row in range(4):
    off = strip0_off + row * 16
    hexbytes = ' '.join(f'{data[off+i]:02x}' for i in range(16))
    print(f"  0x{off:x}: {hexbytes}")

# Check: how many leading zero bits?
first_word = struct.unpack_from('>I', data, strip0_off)[0]
print(f"\nFirst 32-bit word: 0x{first_word:08x} = {first_word:032b}")
if first_word == 0:
    print("  WARNING: starts with 32 zero bits!")
    # Check more
    for i in range(8):
        w = struct.unpack_from('>I', data, strip0_off + i*4)[0]
        print(f"  word {i}: 0x{w:08x}")
else:
    leading_zeros = 0
    tmp = first_word
    while (tmp & 0x80000000) == 0:
        leading_zeros += 1
        tmp <<= 1
    print(f"  Leading zeros: {leading_zeros}")
