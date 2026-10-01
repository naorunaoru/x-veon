import type { DesktopBridge } from '../protocol/bridge';
import { isWorkerPortDelivery } from '../protocol/security';
import { isPortEvent, isPortReply, isPortRequest, type PortEvent, type PortRequest, type StateStamp } from '../protocol/rpc';
/** Install the listener before invoking main: the transferred port may arrive first. */
export function waitForWorkerPort(bridge: DesktopBridge, signal?: AbortSignal): Promise<MessagePort> {
  const requestId = crypto.randomUUID();
  return new Promise((resolve, reject) => {
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const finish = (error?: unknown, port?: MessagePort) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer); window.removeEventListener('message', receive); signal?.removeEventListener('abort', abort);
      if (port) resolve(port); else reject(error);
    };
    const receive = (event: MessageEvent) => {
      if (event.source !== window || event.origin !== location.origin || event.data?.type !== 'xveon-port') return;
      if (!isWorkerPortDelivery(event.data) || event.data.requestId !== requestId || event.ports.length !== 1) {
        for (const port of event.ports) port.close();
        return;
      }
      finish(undefined, event.ports[0]);
    };
    const abort = () => finish(signal?.reason ?? new Error('Worker connection cancelled'));
    if (signal?.aborted) { abort(); return; }
    timer = setTimeout(() => finish(new Error('Worker port connection timed out')), 10_000);
    window.addEventListener('message', receive); signal?.addEventListener('abort', abort, { once: true });
    try { void bridge.requestWorkerPort(requestId).catch(error => finish(error)); }
    catch (error) { finish(error); }
  });
}

type Request = PortRequest extends infer R ? R extends PortRequest ? Omit<R, 'v' | 'rid'> : never : never;
export function createWorkerClient(bridge: DesktopBridge, onFacts: (event: PortEvent) => void, onAcknowledged?: (stamp: StateStamp) => void) {
  let port: MessagePort | null = null, connecting: Promise<MessagePort> | null = null;
  let handshake: AbortController | undefined;
  let generation = 0, rid = 0, stopped: string | null = null;
  const pending = new Map<number, { resolve: () => void; reject: (error: Error) => void }>();
  function disconnect(reason: string) {
    ++generation; handshake?.abort(new Error(reason)); handshake = undefined; port?.close(); port = null; connecting = null;
    for (const request of pending.values()) request.reject(new Error(reason));
    pending.clear();
  }
  function connect(): Promise<MessagePort> {
    if (stopped) return Promise.reject(new Error(stopped));
    if (port) return Promise.resolve(port);
    if (connecting) return connecting;
    const epoch = generation;
    const controller = new AbortController(); handshake = controller;
    const promise = waitForWorkerPort(bridge, controller.signal).then(next => {
      if (epoch !== generation) { next.close(); throw new Error('Worker connection replaced'); }
      port = next;
      next.addEventListener('message', event => {
        if (port !== next) return;
        if (isPortReply(event.data)) {
          const request = pending.get(event.data.rid); if (!request) return;
          pending.delete(event.data.rid);
          if (event.data.ok) { if (event.data.stamp) onAcknowledged?.(event.data.stamp); request.resolve(); } else request.reject(new Error(event.data.error));
        } else if (isPortEvent(event.data)) onFacts(event.data);
      });
      const closed = () => { if (port === next) disconnect('Worker port closed'); };
      next.addEventListener('messageerror', closed); next.addEventListener('close', closed); next.start();
      return next;
    }).finally(() => { if (connecting === promise) connecting = null; if (handshake === controller) handshake = undefined; });
    connecting = promise; return promise;
  }
  // Establish the facts subscription even when the user has not edited anything.
  void connect().catch(() => {});
  return {
    async request(value: Request): Promise<void> {
      const current = await connect();
      if (current !== port) throw new Error('Worker connection replaced');
      const message = { ...value, v: 1, rid: ++rid };
      if (!isPortRequest(message)) throw new Error('Invalid worker request');
      return new Promise<void>((resolve, reject) => {
        const timer = setTimeout(() => { pending.delete(message.rid); reject(new Error('Worker response timed out')); }, 30_000);
        pending.set(message.rid, { resolve: () => { clearTimeout(timer); resolve(); }, reject: error => { clearTimeout(timer); reject(error); } });
        try { current.postMessage(message); } catch (error) {
          pending.delete(message.rid); clearTimeout(timer); reject(error);
        }
      });
    },
    async restart() { stopped = null; disconnect('Worker restarted'); await connect(); },
    stop(reason: string) { stopped = reason; disconnect(reason); },
  };
}
