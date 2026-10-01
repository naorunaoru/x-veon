import type { PhotoEdit } from '@/host';
import type { CfaType, DemosaicMethod } from '@/lib/types';
import { isMethodValidForCfa } from '@/lib/catalog';

export function defaultPhotoEdit(): PhotoEdit {
  return { version: 1, lookPreset: 'default', openDrtOverrides: {}, preProcessOverrides: {}, demosaicMethod: null, model: null };
}

export function isDefaultPhotoEdit(edit: PhotoEdit): boolean {
  return edit.lookPreset === 'default' && edit.demosaicMethod === null && edit.model === null
    && Object.keys(edit.openDrtOverrides).length === 0 && Object.keys(edit.preProcessOverrides).length === 0;
}

/** Resolve the method used for processing without changing the edit. */
export function effectiveMethod(edit: PhotoEdit, defaultMethod: DemosaicMethod, cfaType: CfaType | null): DemosaicMethod {
  if (edit.demosaicMethod) return edit.demosaicMethod;
  return !cfaType || isMethodValidForCfa(defaultMethod, cfaType) ? defaultMethod : 'neural-net';
}
