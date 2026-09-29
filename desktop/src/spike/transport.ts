import { BUILD } from '@/lib/channel';
import { TOTAL_BYTES, CHUNK_BYTES, patternByte } from '../protocol/transfer';
import { connect, response } from './port';
async function transfer(useTransferList: boolean) {
  let source: Uint8Array | null = new Uint8Array(TOTAL_BYTES);
  for (let i = 0; i < source.length; i++) source[i] = patternByte(i);
  const port = await connect();
  const start = performance.now();
  let detached = true;
  try {
    for (let offset = 0; offset < TOTAL_BYTES; offset += CHUNK_BYTES) {
      const buffer = source.slice(
        offset,
        Math.min(offset + CHUNK_BYTES, TOTAL_BYTES),
      ).buffer;
      const ack = await response(port, () =>
        port.postMessage(
          { version: 1, kind: 'chunk', offset, buffer },
          useTransferList ? [buffer] : [],
        ),
      );
      detached &&= buffer.byteLength === 0;
      if (
        ack.kind !== 'received' ||
        ack.received !== Math.min(offset + CHUNK_BYTES, TOTAL_BYTES)
      )
        throw Error('Incorrect receipt');
    }
    source = null;
    const receipt = await response(port, () =>
      port.postMessage({ version: 1, kind: 'finish' }),
    );
    const receiptMs = performance.now() - start;
    const verified = await response(port, () => {});
    if (
      receipt.kind !== 'receipt' ||
      verified.kind !== 'verified' ||
      verified.bytes !== TOTAL_BYTES
    )
      throw Error('Missing verification');
    return {
      bytes: TOTAL_BYTES,
      chunkBytes: CHUNK_BYTES,
      receiptMs,
      verifiedMs: performance.now() - start,
      verifyMs: verified.verifyMs,
      MBps: 400 / (receiptMs / 1000),
      senderChunksDetached: detached,
      workerMemoryAtReceipt: receipt.memory,
      workerMemoryAfterVerification: verified.memory,
    };
  } finally {
    source = null;
    port.close();
  }
}
async function probe(useTransferList: boolean) {
  const port = await connect();
  const buffer = Uint8Array.from({ length: 8 }, (_, i) =>
    patternByte(i),
  ).buffer;
  try {
    const reply = await response(port, () =>
      port.postMessage(
        { version: 1, kind: 'probe', buffer },
        useTransferList ? [buffer] : [],
      ),
    );
    return {
      ok: reply.kind === 'probe' && reply.bytes === 8,
      detached: buffer.byteLength === 0,
      error: null,
    };
  } catch (error) {
    return {
      ok: false,
      detached: buffer.byteLength === 0,
      error: String(error),
    };
  } finally {
    port.close();
  }
}
export async function runTransport() {
  const port = await connect();
  const held = response(port, () =>
    port.postMessage({ version: 1, kind: 'hold' }),
  );
  const restartStart = performance.now();
  const closed = held.then(
    () => ({
      error: 'Unexpected success',
      ms: performance.now() - restartStart,
    }),
    (error) => ({ error: String(error), ms: performance.now() - restartStart }),
  );
  await window.xveon.request('restart');
  const interrupted = await closed;
  if (!interrupted.error.includes('Worker port closed'))
    throw Error('Worker exit did not close active port: ' + interrupted.error);
  port.close();
  const transferListProbe = await probe(true);
  const clonedProbe = await probe(false);
  if (!clonedProbe.ok)
    throw Error('Cloned MessagePort payload failed: ' + clonedProbe.error);
  const useTransferList = transferListProbe.ok;
  const warmup = await transfer(useTransferList),
    runs: Awaited<ReturnType<typeof transfer>>[] = [];
  for (let i = 0; i < 5; i++) runs.push(await transfer(useTransferList));
  const med = (key: 'receiptMs' | 'verifiedMs') =>
    runs.map((r) => r[key]).sort((a, b) => a - b)[2];
  return {
    status: 'PASS',
    commit: BUILD.sha,
    recordedAt: new Date().toISOString(),
    runtime: navigator.userAgent,
    transferListProbe,
    clonedProbe,
    useTransferList,
    restartRejectedActiveRequest: true,
    interrupted,
    warmup,
    runs,
    medianReceiptMs: med('receiptMs'),
    medianVerifiedMs: med('verifiedMs'),
    diagnostics: await window.xveon.request('diagnostics'),
  };
}
