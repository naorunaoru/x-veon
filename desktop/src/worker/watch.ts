import fs from 'node:fs';

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function watchFolder(path: string, onChange: () => void, opts?: { debounceMs?: number }): { close(): void } {
  let watcher: fs.FSWatcher | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let closed = false;
  let reported = false;
  const clear = () => { if (timer) clearTimeout(timer); timer = undefined; };
  const failed = (error: unknown) => {
    if (reported || closed) return;
    reported = true;
    clear();
    watcher?.close();
    console.error(`Folder watcher failed for ${path}: ${message(error)}`);
  };
  try {
    watcher = fs.watch(path, () => {
      if (closed || reported) return;
      clear();
      timer = setTimeout(() => { timer = undefined; if (!closed && !reported) onChange(); }, opts?.debounceMs ?? 200);
    });
    watcher.on('error', failed);
  } catch (error) { failed(error); }
  return {
    close() {
      if (closed) return;
      closed = true;
      clear();
      watcher?.close();
    },
  };
}

export function createWatchHandler(onChange: () => void | Promise<void>, opts?: { debounceMs?: number }) {
  let current: { close(): void } | undefined;
  return (folder: { path: string } | null): void => {
    current?.close();
    current = folder ? watchFolder(folder.path, () => {
      try { void Promise.resolve(onChange()).catch(console.error); }
      catch (error) { console.error(error); }
    }, opts) : undefined;
  };
}
