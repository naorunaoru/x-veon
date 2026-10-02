import type { UnsavedSummary } from '@/host';
type Deps = { window: { on(event: 'close', handler: (e: { preventDefault(): void }) => void): void; destroy(): void }; app: { on(event: 'before-quit', handler: (e: { preventDefault(): void }) => void): void; quit(): void; exit(code?: number): void }; inventory: () => UnsavedSummary[]; requestFlush: () => Promise<UnsavedSummary[]>; confirmQuit: (unsaved: UnsavedSummary[]) => Promise<boolean>; timeoutMs?: number };
export function unsavedQuitMessage(unsaved: UnsavedSummary[]) {
  const summary = unsaved.length === 1
    ? "1 photo has edits that aren't saved."
    : `${unsaved.length} photos have edits that aren't saved.`;
  const photos = unsaved.slice(0, 10).map(edit => `${edit.name} — ${edit.folder?.name ?? 'Unknown folder'}`);
  if (unsaved.length > 10) photos.push(`and ${unsaved.length - 10} more`);
  return `${summary}\n\n${photos.join('\n')}`;
}
export function createCloseGuard(deps: Deps) {
  let allowExit = false, running = false, ending = false;
  async function flush(): Promise<UnsavedSummary[]> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        Promise.resolve().then(deps.requestFlush).catch(() => deps.inventory()),
        new Promise<UnsavedSummary[]>(resolve => { timer = setTimeout(() => resolve(deps.inventory()), deps.timeoutMs ?? 3000); }),
      ]);
    } finally { clearTimeout(timer); }
  }
  function guard(event: { preventDefault(): void }) {
    if (allowExit) return;
    event.preventDefault();
    if (running || ending) return;
    running = true;
    void (async () => {
      const unsaved = await flush();
      if (ending) return;
      if (!unsaved.length || await deps.confirmQuit(unsaved)) {
        if (ending) return;
        allowExit = true; deps.app.quit();
      }
    })().catch(() => {}).finally(() => { running = false; });
  }
  deps.window.on('close', guard); deps.app.on('before-quit', guard);
  return { onSessionEnd() {
    if (ending) return;
    ending = true;
    void flush().finally(() => { allowExit = true; deps.app.exit(0); });
  } };
}
