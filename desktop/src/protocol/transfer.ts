export const TOTAL_BYTES = 400_000_000;
export const CHUNK_BYTES = 64_000_000;
export const patternByte = (offset: number) => (offset * 17 + 31) % 251;
export function createReceiver(
  total = TOTAL_BYTES,
  limit = CHUNK_BYTES,
  verify = true,
) {
  let received = 0;
  return {
    chunk(offset: number, buffer: ArrayBuffer) {
      const bytes = new Uint8Array(buffer);
      if (
        !Number.isSafeInteger(offset) ||
        offset !== received ||
        !bytes.length ||
        bytes.length > limit ||
        received + bytes.length > total
      )
        throw Error('Invalid chunk sequence or size');
      if (verify)
        for (let i = 0; i < bytes.length; i++)
          if (bytes[i] !== patternByte(offset + i))
            throw Error('Content mismatch');
      received += bytes.length;
      return { received };
    },
    finish() {
      if (received !== total) throw Error('Incomplete transfer');
      return received;
    },
  };
}
