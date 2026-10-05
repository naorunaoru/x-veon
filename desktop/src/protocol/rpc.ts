import type { ExportFormat } from '@/lib/types';
import type { FolderRef, LibraryPhoto, PhotoEdit, PhotoFacts, PhotoId } from '@/host';
import { DEMOSAIC_METHODS, MODEL_SIZES } from '@/lib/catalog';
import { LOOK_PRESETS, DEFAULT_PREPROCESS, configFromPreset } from '@/renderer/grading/opendrt-params';
import type { ListingFrame } from './listing';
export type { ListingFrame } from './listing';

export interface StateStamp { worker: string; revision: number }
export interface ListingStamp extends StateStamp { scan: number }
export const EXPORT_CHUNK_BYTES = 64_000_000;
export const MAX_EXPORT_SIDE = 65_535;
export const MAX_EXPORT_PIXELS = 200_000_000;
export interface ExportReceipt { name: string; bytes: number; sha256: string; encodeMs: number }
export type ExportAvailability = { available: true } | { available: false; reason: string };
export type ExportBegin = { v: 1; rid: number; op: 'exportBegin'; job: string; destination: string; format: ExportFormat;
  width: number; height: number; orientation: string; quality: number; peakLuminance: number; planes: 1 | 2 };
export type PortRequest =
  | { v: 1; rid: number; op: 'exportStatus' } | ExportBegin
  | { v: 1; rid: number; op: 'exportChunk'; job: string; plane: 0 | 1; offset: number; data: ArrayBuffer }
  | { v: 1; rid: number; op: 'exportCommit' | 'exportCancel'; job: string }
  | { v: 1; rid: number; op: 'saveEdit'; id: PhotoId; edit: PhotoEdit }
  | { v: 1; rid: number; op: 'saveFacts'; id: PhotoId; facts: PhotoFacts }
  | { v: 1; rid: number; op: 'rescan' };
export type PortReply = { v: 1; rid: number; ok: true; stamp?: StateStamp; receipt?: ExportReceipt; availability?: ExportAvailability } | { v: 1; rid: number; ok: false; error: string };
export type PortEvent = { v: 1; event: 'facts'; activation: string; folder: FolderRef; photos: LibraryPhoto[] };
export type MainToWorker =
  | { v: 1; rid?: number; kind: 'export-destination'; token: string; path: string }
  | { v: 1; kind: 'session'; key: string; cacheDir: string; worker?: string }
  | { v: 1; kind: 'connect' }
  | { v: 1; kind: 'register'; entries: [PhotoId, string][] }
  | { v: 1; kind: 'roots'; realRoots: string[] }
  | { v: 1; rid: number; kind: 'list'; path: string; folderId: string; token: string; activation: string; purpose: 'open' | 'replace' }
  | { v: 1; kind: 'cancel-list'; token: string }
  | { v: 1; rid: number; kind: 'thumbnail'; id: PhotoId }
  | { v: 1; kind: 'watch'; path: string | null; folderId: string | null; activation: string | null };
export type WorkerToMain = ListingFrame | { v: 1; rid: number; kind: 'export-destination'; token: string } | { v: 1; rid: number; kind: 'thumbnail'; path: string | null } | { v: 1; rid: number; kind: 'error'; error: string };

export const MAX_MESSAGE_BYTES = 1_000_000;
type RecordValue = Record<string, unknown>;
const record = (x: unknown): x is RecordValue => x !== null && typeof x === 'object' && !Array.isArray(x);
const string = (x: unknown): x is string => typeof x === 'string';
const number = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
const count = (x: unknown): x is number => Number.isSafeInteger(x) && (x as number) >= 0;
export const isWorkerIdentity = (x: unknown): x is string => typeof x === 'string' && /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(x);
const stateStamp = (x: unknown) => record(x) && isWorkerIdentity(x.worker) && count(x.revision);
const listingStamp = (x: unknown) => stateStamp(x) && count((x as RecordValue).scan);
const id = (x: unknown): x is string => string(x) && /^[A-Za-z0-9_-]{22}$/.test(x);
const jobId = (x: unknown) => typeof x === 'string' && /^[A-Za-z0-9-]{8,64}$/.test(x);
const side = (x: unknown) => Number.isInteger(x) && (x as number) >= 1 && (x as number) <= MAX_EXPORT_SIDE;
const arrayBuffer = (x: unknown): x is ArrayBuffer => Object.prototype.toString.call(x) === '[object ArrayBuffer]';
const receipt = (x: unknown) => record(x) && string(x.name) && count(x.bytes) && string(x.sha256) && /^[a-f0-9]{64}$/i.test(x.sha256) && number(x.encodeMs) && x.encodeMs >= 0;
const availability = (x: unknown) => record(x) && (x.available === true || (x.available === false && string(x.reason)));
const nullableString = (x: unknown) => x === null || string(x);
function array(x: unknown, check: (x: unknown) => boolean): x is unknown[] {
  if (!Array.isArray(x)) return false;
  for (let i = 0; i < x.length; i++) {
    if (!Object.hasOwn(x, i) || !check(x[i])) return false;
  }
  return true;
}
const oneOf = (x: unknown, values: readonly unknown[]) => values.includes(x);
const optional = (x: RecordValue, key: string, check: (v: unknown) => boolean) => x[key] === undefined || check(x[key]);
const fields = (x: RecordValue, names: string[], check: (v: unknown) => boolean) => names.every(k => check(x[k]));
const method = (x: unknown) => DEMOSAIC_METHODS.some(m => m.id === x);
const cfa = (x: unknown) => oneOf(x, ['xtrans', 'bayer']);
const model = (x: unknown) => record(x) && oneOf(x.size, MODEL_SIZES) && string(x.sha256);
const folder = (x: unknown): x is FolderRef => record(x) && string(x.id) && string(x.name);
const purpose = (x: unknown) => oneOf(x, ['open', 'replace']);
const defaults = configFromPreset('default');
function overrides(x: unknown, shape: object): boolean {
  return record(x) && Object.entries(x).every(([key, value]) => Object.hasOwn(shape, key)
    && (typeof (shape as RecordValue)[key] === 'boolean' ? typeof value === 'boolean' : number(value)));
}
function edit(x: unknown): x is PhotoEdit {
  return record(x) && x.version === 1 && string(x.lookPreset) && Object.hasOwn(LOOK_PRESETS, x.lookPreset)
    && overrides(x.openDrtOverrides, defaults) && overrides(x.preProcessOverrides, DEFAULT_PREPROCESS)
    && (x.demosaicMethod === null || method(x.demosaicMethod)) && (x.model === null || model(x.model));
}
function result(x: unknown): boolean {
  if (!record(x) || !record(x.exportData) || !record(x.metadata)) return false;
  const e = x.exportData, m = x.metadata;
  return fields(e, ['width', 'height'], number) && string(e.orientation)
    && (e.xyzToCam === null || array(e.xyzToCam, number)) && array(e.wbCoeffs, number)
    && optional(e, 'camToXyz', v => array(v, number))
    && fields(m, ['make', 'model', 'backend', 'lensModel'], string)
    && fields(m, ['width', 'height', 'tileCount', 'inferenceTime', 'exposureBias', 'focalLength', 'fNumber', 'colorTemp', 'tint'], number)
    && optional(m, 'modelSize', v => oneOf(v, MODEL_SIZES)) && optional(m, 'modelIdentity', model)
    && optional(m, 'cfaType', cfa) && optional(m, 'modelNote', nullableString);
}
function coefficients(x: unknown, required: string[], optionalNames: string[]): boolean {
  return record(x) && string(x.model) && fields(x, required, number) && optionalNames.every(k => optional(x, k, number));
}
function lens(x: unknown): boolean {
  return record(x) && string(x.lensModel) && string(x.mount) && number(x.cropfactor)
    && array(x.distortion, v => coefficients(v, ['focal'], ['k1', 'k2', 'a', 'b', 'c']))
    && array(x.tca, v => coefficients(v, ['focal'], ['vr', 'vb', 'br', 'cr', 'bb', 'cb', 'kr', 'kb']))
    && array(x.vignetting, v => coefficients(v, ['focal', 'aperture', 'distance', 'k1', 'k2', 'k3'], []));
}
function facts(x: unknown): x is PhotoFacts {
  return record(x) && (x.cfaType === null || cfa(x.cfaType))
    && (x.metadata === null || (record(x.metadata) && fields(x.metadata, ['camera', 'lensModel'], string) && fields(x.metadata, ['focalLength', 'fNumber'], number)))
    && (x.resultMeta === null || result(x.resultMeta)) && (x.resultMethod === null || method(x.resultMethod))
    && (x.lensProfile === null || lens(x.lensProfile)) && oneOf(x.status, ['queued', 'done', 'error']) && nullableString(x.error);
}
function photo(x: unknown): x is LibraryPhoto {
  return record(x) && id(x.id) && string(x.name) && string(x.originalName) && number(x.fileSize) && x.fileSize >= 0
    && (x.sourceVersion === undefined || (typeof x.sourceVersion === 'string' && /^[a-f0-9]{64}$/.test(x.sourceVersion)))
    && nullableString(x.thumbnailUrl) && edit(x.edit) && facts(x.facts)
    && oneOf(x.editing, ['saved', 'session', 'view-only']) && nullableString(x.editingNote);
}
const photos = (x: unknown) => array(x, photo) && x.length <= 250;
const registry = (x: unknown, max: number) => array(x, e => Array.isArray(e) && e.length === 2 && id(e[0]) && string(e[1])) && x.length <= max;
/** UTF-8 bytes, so this works identically in the sandboxed window and Node. */
function envelope(x: unknown): x is RecordValue {
  if (!record(x) || x.v !== 1) return false;
  try { return new TextEncoder().encode(JSON.stringify(x)).byteLength <= MAX_MESSAGE_BYTES; }
  catch { return false; }
}
function listing(x: RecordValue): boolean {
  if (!string(x.token)) return false;
  switch (x.kind) {
    case 'listing-begin': return string(x.activation) && folder(x.folder) && count(x.total) && purpose(x.purpose) && optional(x, 'stamp', listingStamp);
    case 'listing-batch': return count(x.seq) && photos(x.photos) && optional(x, 'registry', v => registry(v, 250));
    case 'listing-end': return count(x.total);
    default: return false;
  }
}
export function isListingFrame(x: unknown): x is ListingFrame { return envelope(x) && listing(x); }
export function isPortRequest(x: unknown): x is PortRequest {
  if (!envelope(x) || !Number.isSafeInteger(x.rid)) return false;
  switch (x.op) {
    case 'saveEdit': return id(x.id) && edit(x.edit);
    case 'saveFacts': return id(x.id) && facts(x.facts);
    case 'rescan': case 'exportStatus': return true;
    case 'exportBegin': return jobId(x.job) && jobId(x.destination) && oneOf(x.format, ['avif', 'jpeg-hdr', 'tiff'])
      && side(x.width) && side(x.height) && (x.width as number) * (x.height as number) <= MAX_EXPORT_PIXELS
      && typeof x.orientation === 'string' && x.orientation.length <= 32
      && Number.isInteger(x.quality) && (x.quality as number) >= 1 && (x.quality as number) <= 100
      && number(x.peakLuminance) && x.peakLuminance > 0 && x.peakLuminance <= 10_000
      && x.planes === (x.format === 'jpeg-hdr' ? 2 : 1);
    case 'exportChunk': return jobId(x.job) && (x.plane === 0 || x.plane === 1)
      && Number.isSafeInteger(x.offset) && (x.offset as number) >= 0 && (x.offset as number) % 4 === 0
      && arrayBuffer(x.data) && x.data.byteLength >= 4 && x.data.byteLength <= EXPORT_CHUNK_BYTES && x.data.byteLength % 4 === 0;
    case 'exportCommit': case 'exportCancel': return jobId(x.job);
    default: return false;
  }
}
export function isPortReply(x: unknown): x is PortReply {
  return envelope(x) && Number.isSafeInteger(x.rid) && ((x.ok === true && optional(x, 'stamp', stateStamp) && optional(x, 'receipt', receipt) && optional(x, 'availability', availability)) || (x.ok === false && string(x.error)));
}
export function isPortEvent(x: unknown): x is PortEvent {
  return envelope(x) && x.event === 'facts' && string(x.activation) && folder(x.folder) && photos(x.photos);
}
export function isMainToWorker(x: unknown): x is MainToWorker {
  if (!envelope(x)) return false;
  switch (x.kind) {
    case 'session': return string(x.key) && /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(x.key) && x.key.length > 0 && string(x.cacheDir) && optional(x, 'worker', isWorkerIdentity);
    case 'connect': return true;
    case 'export-destination': return jobId(x.token) && string(x.path) && x.path.length > 0 && optional(x, 'rid', Number.isSafeInteger);
    case 'register': return registry(x.entries, 1000);
    case 'roots': return array(x.realRoots, string);
    case 'list': return Number.isSafeInteger(x.rid) && string(x.path) && string(x.folderId) && string(x.token) && string(x.activation) && purpose(x.purpose);
    case 'cancel-list': return string(x.token);
    case 'thumbnail': return Number.isSafeInteger(x.rid) && id(x.id);
    case 'watch': return (x.path === null && x.folderId === null && x.activation === null) || (string(x.path) && string(x.folderId) && string(x.activation));
    default: return false;
  }
}
export function isWorkerToMain(x: unknown): x is WorkerToMain {
  if (!envelope(x)) return false;
  return listing(x) || (Number.isSafeInteger(x.rid) && ((x.kind === 'export-destination' && jobId(x.token)) || (x.kind === 'thumbnail' && nullableString(x.path)) || (x.kind === 'error' && string(x.error))));
}
