import { useEffect } from 'react';
import { initApp } from '@/app/services/bootstrap';
import { startPersistence } from '@/app/services/persistence';

/** Mount once near the root: starts persistence and the app initialisation. */
export function useBootstrap(): void {
  useEffect(() => {
    const stopPersistence = startPersistence();
    const signal = { cancelled: false };
    initApp(signal);
    return () => {
      signal.cancelled = true;
      stopPersistence();
    };
  }, []);
}
