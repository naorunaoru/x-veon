import { useAppStore } from '@/store';

export function Header() {
  const initialized = useAppStore((s) => s.initialized);
  const initError = useAppStore((s) => s.initError);
  const backend = useAppStore((s) => s.backend);
  const hdrSupported = useAppStore((s) => s.hdrSupported);
  const modelMeta = useAppStore((s) => s.modelMeta);

  return (
    <header className="px-4 py-3 border-b border-border">
      <h1 className="text-base font-semibold">X-veon: neural demosaic for X-Trans sensors</h1>
      <p className="text-xs text-muted-foreground mt-0.5">
        {initError
          ? `Init failed: ${initError}`
          : initialized
            ? `${backend}${hdrSupported ? ', HDR' : ''}${modelMeta.epoch ? ` · epoch ${modelMeta.epoch}` : ''}${modelMeta.best_val_psnr ? ` · ${modelMeta.best_val_psnr} dB` : ''}`
            : 'Loading model and WASM\u2026'}
      </p>
    </header>
  );
}
