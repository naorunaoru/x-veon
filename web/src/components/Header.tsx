import { useAppStore } from '@/store';

export function Header() {
  const initialized = useAppStore((s) => s.initialized);
  const initError = useAppStore((s) => s.initError);
  const backend = useAppStore((s) => s.backend);
  const displayHdr = useAppStore((s) => s.displayHdr);

  return (
    <header className="px-4 py-3 border-b border-border">
      <h1 className="text-base font-semibold">X-veon RAW processor</h1>
      <p className="text-xs text-muted-foreground mt-0.5">
        {initError
          ? `Init failed: ${initError}`
          : initialized
            ? `${backend}${displayHdr ? ', HDR' : ''}`
            : 'Loading models and WASM\u2026'}
      </p>
    </header>
  );
}
