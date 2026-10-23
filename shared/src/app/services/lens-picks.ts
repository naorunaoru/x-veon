import { loadLensProfile } from '@/app/lens/lensfun';
import { useAppStore } from '@/app/store';
import { lensPickKey } from '@/app/store/lens';

const loading = new Set<string>();

/**
 * Loads a hand-picked lensfun lens into the session picks, once. A failed or empty load can be retried by calling again.
 * The profile is fitted to the camera of the photo it is picked for.
 */
export function ensureLensPick(fileId: string, choice: { file: string; model: string }): void {
  const key = lensPickKey(choice.file, choice.model);
  const state = useAppStore.getState();
  const file = state.files.find((f) => f.id === fileId);
  if (!file || key in state.lensPicks || loading.has(key)) return;
  loading.add(key);
  loadLensProfile(choice, file.metadata?.camera ?? '')
    .then((profile) => { if (profile) useAppStore.getState().setLensPick(key, profile); })
    .catch((error) => console.warn('Lens load failed:', error))
    .finally(() => loading.delete(key));
}
