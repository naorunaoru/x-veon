import { defaultPhotoEdit } from '@/app/photo-edit';
import type { PhotoEdit, Rating } from '@/host';
import type { DemosaicMethod, LookPreset, ModelSize } from '@/lib/types';
import { DEMOSAIC_METHODS, MODEL_SIZES } from '@/lib/catalog';
import { DEFAULT_LENS_EDIT, effectiveLensEdit } from '@/lib/lens/edit';
import type { LensEdit, LensManual } from '@/lib/lens/edit';
import { configFromPreset, LOOK_PRESETS } from '@/renderer/grading/opendrt-params';

export const XVEON_NS = 'https://naorunaoru.github.io/x-veon/ns/1.0/';
export const XMP_NS = 'http://ns.adobe.com/xap/1.0/';
/** Namespaced property name → value, e.g. 'xveon:Look' → 'umbra', 'xmp:Rating' → '3'. */
export type SidecarProperties = Record<string, string>;

const XVEON = 'xveon:';
const RATING = 'xmp:Rating';
const PRE: Record<string, keyof PhotoEdit['preProcessOverrides']> = {
  Exposure: 'exposure', WbTemp: 'wb_temp', WbTint: 'wb_tint', Sharpen: 'sharpen_amount',
};
const LENS_FLAGS: Record<string, 'distortion' | 'ca' | 'vignetting'> = {
  LensDistortion: 'distortion', LensCa: 'ca', LensVignetting: 'vignetting',
};
const LENS_MANUAL: Record<string, keyof LensManual> = {
  LensManualDistortion: 'distortion', LensManualCaRed: 'caRed', LensManualCaBlue: 'caBlue',
  LensManualVignetting: 'vignetting', LensManualMidpoint: 'midpoint',
};
const DRT_DEFAULT = configFromPreset('default');
const DRT_KEYS = new Set(Object.keys(DRT_DEFAULT));
const LOOKS = new Set<string>(Object.keys(LOOK_PRESETS));
const METHODS = new Set<string>(DEMOSAIC_METHODS.map((method) => method.id));
const SIZES = new Set<string>(MODEL_SIZES);

export function isRating(value: unknown): value is Rating {
  return typeof value === 'number' && Number.isInteger(value) && value >= -1 && value <= 5;
}

function lensName(name: string): boolean {
  return name === 'LensSource' || name === 'LensFile' || name === 'LensModel'
    || Object.hasOwn(LENS_FLAGS, name) || Object.hasOwn(LENS_MANUAL, name);
}
function knownXveonName(name: string): boolean {
  return name === 'SchemaVersion' || name === 'Look' || name === 'Demosaic' || name === 'ModelSize' || name === 'ModelHash'
    || lensName(name) || Object.hasOwn(PRE, name) || (name.startsWith('Odrt_') && DRT_KEYS.has(name.slice(5)));
}
export function isKnownProperty(name: string): boolean {
  return name === RATING || (name.startsWith(XVEON) && knownXveonName(name.slice(XVEON.length)));
}
export function unknownProperties(properties: SidecarProperties): SidecarProperties {
  return Object.fromEntries(Object.entries(properties).filter(([name]) => !isKnownProperty(name)));
}

function numberValue(name: string, raw: string): number {
  const trimmed = raw.trim();
  if (!trimmed || !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed)) throw new Error(`${name}: invalid number`);
  const value = Number(trimmed);
  if (!Number.isFinite(value)) throw new Error(`${name}: non-finite number`);
  return value;
}
function booleanValue(name: string, raw: string): boolean {
  if (raw === 'True') return true;
  if (raw === 'False') return false;
  throw new Error(`${name}: invalid boolean`);
}
function ratingValue(raw: string | undefined): Rating {
  if (raw === undefined) return 0;
  const trimmed = raw.trim();
  const value = Number(trimmed);
  if (!/^-?\d+$/.test(trimmed) || !isRating(value)) throw new Error(`${RATING}: invalid rating`);
  return value;
}
function readLens(values: Map<string, string>): LensEdit {
  const lens = effectiveLensEdit({});
  const source = values.get('LensSource');
  if (source === 'auto' || source === 'manual') lens.source = source;
  else if (source === 'lensfun') {
    const file = values.get('LensFile');
    const model = values.get('LensModel');
    if (!file || !model) throw new Error('LensSource: lensfun needs LensFile and LensModel');
    lens.source = { lensfun: { file, model } };
  } else if (source !== undefined) throw new Error('LensSource: invalid source');
  for (const [name, key] of Object.entries(LENS_FLAGS)) {
    const raw = values.get(name);
    if (raw !== undefined) lens[key] = booleanValue(name, raw);
  }
  for (const [name, key] of Object.entries(LENS_MANUAL)) {
    const raw = values.get(name);
    if (raw !== undefined) lens.manual[key] = numberValue(name, raw);
  }
  return lens;
}

export function decodeSidecar(properties: SidecarProperties): { edit: PhotoEdit; rating: Rating; schemaVersion: number; unknown: SidecarProperties } {
  const values = new Map<string, string>();
  for (const [name, value] of Object.entries(properties)) if (name.startsWith(XVEON)) values.set(name.slice(XVEON.length), value);
  const edit = defaultPhotoEdit();
  const rawVersion = values.get('SchemaVersion');
  const schemaVersion = rawVersion === undefined ? 1 : numberValue('SchemaVersion', rawVersion);
  if (!Number.isInteger(schemaVersion) || schemaVersion < 1) throw new Error('SchemaVersion: invalid version');
  const look = values.get('Look');
  if (look !== undefined) {
    if (!LOOKS.has(look)) throw new Error('Look: invalid preset');
    edit.lookPreset = look as LookPreset;
  }
  const method = values.get('Demosaic');
  if (method !== undefined) {
    if (!METHODS.has(method)) throw new Error('Demosaic: invalid method');
    edit.demosaicMethod = method as DemosaicMethod;
  }
  for (const [name, key] of Object.entries(PRE)) {
    const raw = values.get(name);
    if (raw !== undefined) edit.preProcessOverrides[key] = numberValue(name, raw);
  }
  for (const [name, raw] of values) {
    if (!name.startsWith('Odrt_')) continue;
    const key = name.slice(5) as keyof PhotoEdit['openDrtOverrides'];
    if (!DRT_KEYS.has(key)) continue;
    const value = typeof DRT_DEFAULT[key] === 'boolean' ? booleanValue(name, raw) : numberValue(name, raw);
    Object.assign(edit.openDrtOverrides, { [key]: value });
  }
  const size = values.get('ModelSize');
  const hash = values.get('ModelHash');
  if (size !== undefined || hash !== undefined) {
    if (!size || !SIZES.has(size)) throw new Error('ModelSize: invalid size');
    if (!hash) throw new Error('ModelHash: missing hash');
    edit.model = { size: size as ModelSize, sha256: hash };
  }
  if ([...values.keys()].some(lensName)) edit.lens = readLens(values);
  return { edit, rating: ratingValue(properties[RATING]), schemaVersion, unknown: unknownProperties(properties) };
}

function writeNumber(name: string, value: number): string {
  if (!Number.isFinite(value)) throw new Error(`${name}: non-finite number`);
  return String(value);
}
function editValues(edit: PhotoEdit): Map<string, string> {
  const result = new Map<string, string>();
  if (edit.lookPreset !== 'default') result.set('Look', edit.lookPreset);
  if (edit.demosaicMethod !== null) result.set('Demosaic', edit.demosaicMethod);
  for (const [name, key] of Object.entries(PRE)) {
    if (Object.hasOwn(edit.preProcessOverrides, key)) {
      const value = edit.preProcessOverrides[key];
      if (value !== undefined) result.set(name, writeNumber(name, value));
    }
  }
  for (const [key, value] of Object.entries(edit.openDrtOverrides)) {
    if (value === undefined) continue;
    if (!DRT_KEYS.has(key)) throw new Error(`Odrt_${key}: unknown override`);
    const name = `Odrt_${key}`;
    if (typeof value === 'number') result.set(name, writeNumber(name, value));
    else if (typeof value === 'boolean') result.set(name, value ? 'True' : 'False');
    else throw new Error(`${name}: invalid value`);
  }
  if (edit.demosaicMethod === 'neural-net' && edit.model !== null) {
    result.set('ModelSize', edit.model.size);
    result.set('ModelHash', edit.model.sha256);
  }
  const lens = effectiveLensEdit(edit);
  if (lens.source !== 'auto') result.set('LensSource', typeof lens.source === 'string' ? lens.source : 'lensfun');
  if (typeof lens.source === 'object') {
    result.set('LensFile', lens.source.lensfun.file);
    result.set('LensModel', lens.source.lensfun.model);
  }
  for (const [name, key] of Object.entries(LENS_FLAGS)) if (!lens[key]) result.set(name, 'False');
  for (const [name, key] of Object.entries(LENS_MANUAL)) {
    if (lens.manual[key] !== DEFAULT_LENS_EDIT.manual[key]) result.set(name, writeNumber(name, lens.manual[key]));
  }
  return result;
}

export function encodeSidecar(input: { edit: PhotoEdit; rating: Rating }, unknown: SidecarProperties = {}): SidecarProperties {
  const known: SidecarProperties = {};
  for (const [name, value] of editValues(input.edit)) known[`${XVEON}${name}`] = value;
  if (input.rating !== 0) known[RATING] = String(input.rating);
  const result = { ...known, ...unknown };
  const versioned = Object.keys(result).some((name) => name.startsWith(XVEON));
  return versioned ? { [`${XVEON}SchemaVersion`]: '1', ...result } : result;
}
