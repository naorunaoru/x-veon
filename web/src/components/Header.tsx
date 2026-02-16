import { useAppStore } from '@/store';

export function Header() {
  const initialized = useAppStore((s) => s.initialized);
  const initError = useAppStore((s) => s.initError);
  const backend = useAppStore((s) => s.backend);
  const displayHdr = useAppStore((s) => s.displayHdr);
  const modelMeta = useAppStore((s) => s.modelMeta);

  return (
    <header className="px-4 py-3 border-b border-border">
      <h1 className="text-base font-semibold">X-veon: neural demosaic</h1>
      <p className="text-xs text-muted-foreground mt-0.5">
        {initError
          ? `Init failed: ${initError}`
          : initialized
            ? `${backend}${displayHdr ? ', HDR' : ''}${modelMeta.epoch ? ` · epoch ${modelMeta.epoch}` : ''}${modelMeta.best_val_psnr ? ` · ${modelMeta.best_val_psnr} dB` : ''}`
            : 'Loading models and WASM\u2026'}
      </p>
    </header>
  );
}
