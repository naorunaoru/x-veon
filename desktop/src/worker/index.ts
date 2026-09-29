import type {} from 'electron';
import { createReceiver, TOTAL_BYTES } from '../protocol/transfer';
process.parentPort.on('message', (event) => {
  if (
    event.data?.version !== 1 ||
    event.data.kind !== 'connect' ||
    event.ports.length !== 1
  )
    return;
  const port = event.ports[0];
  let receiver = createReceiver(TOTAL_BYTES, undefined, false);
  let chunks: { offset: number; buffer: ArrayBuffer }[] = [];
  port.on('message', (event) => {
    try {
      const data = event.data;
      if (data === null) throw Error('Port payload deserialized as null');
      if (data?.version !== 1) throw Error('Wrong transfer version');
      if (data.kind === 'probe') {
        const check = createReceiver(8, 8);
        check.chunk(0, data.buffer);
        check.finish();
        port.postMessage({ kind: 'probe', bytes: 8 });
        return;
      }
      if (data.kind === 'hold') return; // Spike probe: a request interrupted by worker exit.
      if (data.kind === 'chunk' && data.buffer instanceof ArrayBuffer) {
        const ack = receiver.chunk(data.offset, data.buffer);
        chunks.push({ offset: data.offset, buffer: data.buffer });
        port.postMessage({ kind: 'received', ...ack });
      } else if (data.kind === 'finish') {
        receiver.finish();
        const memory = process.memoryUsage();
        port.postMessage({ kind: 'receipt', bytes: TOTAL_BYTES, memory });
        setImmediate(() => {
          try {
            const verifier = createReceiver();
            const start = performance.now();
            for (const chunk of chunks)
              verifier.chunk(chunk.offset, chunk.buffer);
            verifier.finish();
            chunks = [];
            receiver = createReceiver(TOTAL_BYTES, undefined, false);
            port.postMessage({
              kind: 'verified',
              bytes: TOTAL_BYTES,
              verifyMs: performance.now() - start,
              memory: process.memoryUsage(),
            });
          } catch (error) {
            port.postMessage({ kind: 'error', error: String(error) });
            port.close();
          }
        });
      } else throw Error('Invalid transfer message');
    } catch (error) {
      chunks = [];
      port.postMessage({ kind: 'error', error: String(error) });
      port.close();
    }
  });
  port.on('close', () => {
    chunks = [];
  });
  port.start();
});
