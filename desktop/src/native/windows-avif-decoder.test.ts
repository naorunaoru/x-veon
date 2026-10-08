import { createHash } from 'node:crypto';
import { expect, it, vi } from 'vitest';
import { decodeWindowsAvif, knownGoodAvif, type Decode, type DecoderCause } from './windows-avif-decoder';

const unsupported: DecoderCause = { type: 'System.NotSupportedException', hResult: -2146233067, message: 'No imaging component suitable' };
const missing: DecoderCause = { type: 'System.Runtime.InteropServices.COMException', hResult: -2003292336, message: 'Component not found' };
const wrapper: DecoderCause = { type: 'System.Reflection.TargetInvocationException', hResult: -2146232828, message: 'Invocation failed' };
const success = { centers: [[0, 0, 0]] };

it('pins the independent known-good AVIF so encoder changes cannot redefine the capability probe', () => {
  expect(createHash('sha256').update(knownGoodAvif).digest('hex')).toBe('0c29cdc227d56730bef68a068b3116ead3bfcd10aa6a9572a2c30e92500b91d7');
});

it.each([unsupported, missing])('detects a missing codec only from the fixed reference: $type', cause => {
  const decode = vi.fn<Decode>().mockReturnValue({ error: [wrapper, cause] });
  expect(decodeWindowsAvif('reference.avif', 'generated.avif', decode)).toEqual({ unavailable: true, causes: [wrapper, cause] });
  expect(decode.mock.calls).toEqual([['reference.avif']]);
});

it('does not hide unexpected reference failures', () => {
  const error = { type: 'System.IO.IOException', hResult: -2146232800, message: 'Read failed' };
  const decode = vi.fn<Decode>().mockReturnValue({ error: [error] });
  expect(() => decodeWindowsAvif('reference.avif', 'generated.avif', decode)).toThrow('Read failed');
});

it.each([unsupported, missing])('never skips generated-file failures after the reference decodes: $type', cause => {
  const decode = vi.fn<Decode>().mockReturnValueOnce(success).mockReturnValueOnce({ error: [cause] });
  expect(() => decodeWindowsAvif('reference.avif', 'generated.avif', decode)).toThrow('generated.avif');
  expect(decode.mock.calls).toEqual([['reference.avif'], ['generated.avif']]);
});

it('returns generated pixels unchanged for the existing color assertions', () => {
  const pixels = { centers: [[1, .2, .3], [.4, .5, .6]] };
  const decode = vi.fn<Decode>().mockReturnValueOnce(success).mockReturnValueOnce(pixels);
  expect(decodeWindowsAvif('reference.avif', 'generated.avif', decode)).toBe(pixels);
});

it('propagates process failures and timeouts instead of classifying them as missing codecs', () => {
  const decode = vi.fn<Decode>().mockImplementation(() => { throw new Error('ETIMEDOUT'); });
  expect(() => decodeWindowsAvif('reference.avif', 'generated.avif', decode)).toThrow('ETIMEDOUT');
});
