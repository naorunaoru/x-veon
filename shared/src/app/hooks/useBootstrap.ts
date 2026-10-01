import { useEffect } from 'react';
import { startLibraryWatching } from '@/app/services/library';
import { initApp, startHostCoordination } from '@/app/services/bootstrap';
import { startPersistence } from '@/app/services/persistence';

/** Mount once near the root: starts persistence and the app initialisation. */
export function useBootstrap(): void {
  useEffect(() => {
    const stopWatching = startLibraryWatching();
    const stopPersistence = startPersistence();
    const stopCoordination = startHostCoordination();
    const signal = { cancelled: false };
    initApp(signal);
    return () => {
      signal.cancelled = true;
      stopCoordination();
      stopPersistence();
      stopWatching();
    };
  }, []);
}
