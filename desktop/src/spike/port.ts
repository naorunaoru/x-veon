/** One request in flight: bounds buffer ownership and makes disconnects explicit. */
export function response(
  port: MessagePort,
  send: () => void,
  timeoutMs = 30_000,
): Promise<any> {
  return new Promise((resolve, reject) => {
    const finish = (error?: Error, value?: unknown) => {
      clearTimeout(timer);
      port.removeEventListener('message', message);
      port.removeEventListener('close', closed);
      port.removeEventListener('messageerror', closed);
      if (error) reject(error);
      else resolve(value);
    };
    const message = (event: MessageEvent) =>
      event.data?.kind === 'error'
        ? finish(Error(event.data.error))
        : finish(undefined, event.data);
    const closed = () => finish(Error('Worker port closed'));
    const timer = setTimeout(() => {
      finish(Error('Worker response timed out'));
      port.close();
    }, timeoutMs);
    port.addEventListener('message', message);
    port.addEventListener('close', closed);
    port.addEventListener('messageerror', closed);
    port.start();
    try {
      send();
    } catch (error) {
      finish(error instanceof Error ? error : Error(String(error)));
    }
  });
}
export function connect(): Promise<MessagePort> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      window.removeEventListener('message', receive);
      reject(Error('Port connection timed out'));
    }, 10_000);
    const receive = (event: MessageEvent) => {
      if (
        event.source !== window ||
        event.origin !== location.origin ||
        event.data?.type !== 'xveon-port' ||
        event.data.version !== 1 ||
        event.ports.length !== 1
      )
        return;
      clearTimeout(timer);
      window.removeEventListener('message', receive);
      resolve(event.ports[0]);
    };
    window.addEventListener('message', receive);
    void window.xveon.request('connect').catch((error) => {
      clearTimeout(timer);
      window.removeEventListener('message', receive);
      reject(error);
    });
  });
}
