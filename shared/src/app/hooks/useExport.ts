import { getHost } from '@/app/services/host';
import { useEffect, useState } from 'react';
export function useExport() {
  const [unavailableReason, setUnavailableReason] = useState<string | null>(null);
  const [exportAvailable, setExportAvailable] = useState(false);
  useEffect(() => {
    let cancelled = false;
    void getHost()
      .exporter.status()
      .then((status) => {
        if (cancelled) return;
        setExportAvailable(status.available);
        if (!status.available) setUnavailableReason(status.reason);
      })
      .catch((error) => {
        if (!cancelled) setUnavailableReason(error instanceof Error ? error.message : String(error));
      });
    return () => {
      cancelled = true;
    };
  }, []);
  return { exportAvailable, unavailableReason };
}
