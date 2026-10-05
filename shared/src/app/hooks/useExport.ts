import { getHost } from '@/app/services/host';
import { useCallback, useEffect, useState } from 'react';
export function useExport() {
  const [unavailableReason, setUnavailableReason] = useState<string | null>(null);
  const [exportAvailable, setExportAvailable] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const retry = useCallback(() => setAttempt(value => value + 1), []);
  useEffect(() => {
    let cancelled = false;
    setExportAvailable(false);
    setUnavailableReason(null);
    void getHost()
      .exporter.status()
      .then((status) => {
        if (cancelled) return;
        setExportAvailable(status.available);
        setUnavailableReason(status.available ? null : status.reason);
      })
      .catch((error) => {
        if (!cancelled) setUnavailableReason(error instanceof Error ? error.message : String(error));
      });
    return () => {
      cancelled = true;
    };
  }, [attempt]);
  return { exportAvailable, unavailableReason, retry };
}
