import { expect, it } from 'vitest';
import {
  createReceiver,
  patternByte,
  TOTAL_BYTES,
  CHUNK_BYTES,
} from './transfer';
it('checks chunk sequence, exact length, content and final totals', () => {
  const receiver = createReceiver(10, 4);
  for (let offset = 0; offset < 10; offset += 4) {
    const bytes = Uint8Array.from(
      { length: Math.min(4, 10 - offset) },
      (_, i) => patternByte(offset + i),
    );
    expect(receiver.chunk(offset, bytes.buffer)).toEqual({
      received: Math.min(offset + 4, 10),
    });
  }
  expect(receiver.finish()).toBe(10);
  expect(TOTAL_BYTES).toBe(400_000_000);
  expect(CHUNK_BYTES).toBe(64_000_000);
});
it('rejects bad data and incomplete or oversized transfers', () => {
  expect(() => createReceiver(10, 4).chunk(1, new ArrayBuffer(4))).toThrow();
  expect(() => createReceiver(10, 4).chunk(0, new ArrayBuffer(5))).toThrow();
  expect(() => createReceiver(10, 4).chunk(0, new ArrayBuffer(4))).toThrow();
  expect(() => createReceiver(10, 4).finish()).toThrow();
  expect(() => createReceiver(10, 4).chunk(0, new ArrayBuffer(0))).toThrow();
});
