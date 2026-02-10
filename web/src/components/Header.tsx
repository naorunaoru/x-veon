import { useAppStore } from '@/store';

export function Header() {
  const initialized = useAppStore((s) => s.initialized);
  const initError = useAppStore((s) => s.initError);
  const backend = useAppStore((s) => s.backend);
  const hdrSupported = useAppStore((s) => s.hdrSupported);

  return (
    <header className="px-4 py-3 border-b border-border">
      <h1 className="text-base font-semibold">X-Trans Demosaic</h1>
      <p className="text-xs text-muted-foreground mt-0.5">
        {initError
          ? `Init failed: ${initError}`
          : initialized
            ? `${backend}${hdrSupported ? ', HDR' : ''}`
            : 'Loading model and WASM\u2026'}
      </p>
    </header>
  );
}
