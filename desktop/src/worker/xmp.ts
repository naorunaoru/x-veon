import { DOMParser, XMLSerializer } from '@xmldom/xmldom';
import type { Document, Element, Node } from '@xmldom/xmldom';
import { defaultPhotoEdit } from '@/app/photo-edit';
import type { PhotoEdit } from '@/host';
import type { DemosaicMethod, LookPreset, ModelSize } from '@/lib/types';
import { configFromPreset } from '@/renderer/grading/opendrt-params';

export const XVEON_NS = 'https://naorunaoru.github.io/x-veon/ns/1.0/';
const RDF_NS = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#';
const XMLNS_NS = 'http://www.w3.org/2000/xmlns/';
const NEW_FILE = `<x:xmpmeta xmlns:x="adobe:ns:meta/">\n <rdf:RDF xmlns:rdf="${RDF_NS}">\n  <rdf:Description rdf:about="" xmlns:xveon="${XVEON_NS}"/>\n </rdf:RDF>\n</x:xmpmeta>`;
const PRE: Record<string, keyof PhotoEdit['preProcessOverrides']> = {
  Exposure: 'exposure', WbTemp: 'wb_temp', WbTint: 'wb_tint', Sharpen: 'sharpen_amount',
};
const DRT_DEFAULT = configFromPreset('default');
const DRT_KEYS = new Set(Object.keys(DRT_DEFAULT));
const LOOKS = new Set<LookPreset>(['opendrt-v1-default', 'opendrt-v1-colorful', 'opendrt-v1-umbra', 'opendrt-v1-base', 'default', 'colorful', 'umbra', 'base', 'flat', 'low-contrast', 'medium-contrast', 'aces-2', 'marvelous']);
const METHODS = new Set<DemosaicMethod>(['neural-net', 'bilinear', 'markesteijn3', 'markesteijn1', 'dht', 'ahd', 'ppg', 'mhc']);
const SIZES = new Set<ModelSize>(['S', 'M', 'L']);

export type SidecarState =
  | { kind: 'absent' }
  | { kind: 'ok'; edit: PhotoEdit }
  | { kind: 'newer'; edit: PhotoEdit; schemaVersion: number }
  | { kind: 'unreadable'; reason: string };

type Parsed = { doc: Document; rdf: Element; description: Element | null; values: Map<string, string> };

function parse(text: string): Parsed {
  const errors: string[] = [];
  const doc = new DOMParser({ onError: (_level, message) => { errors.push(message); } }).parseFromString(text.replace(/^\uFEFF/, ''), 'application/xml');
  if (errors.length) throw new Error(`XML: ${errors[0]}`);
  const rdf = doc.getElementsByTagNameNS(RDF_NS, 'RDF').item(0);
  if (!rdf) throw new Error('Missing rdf:RDF');
  const descriptions = Array.from(rdf.getElementsByTagNameNS(RDF_NS, 'Description'));
  const description = descriptions.find(hasXveonProperty)
    ?? descriptions.find((element) => element.getAttributeNS(RDF_NS, 'about') === '')
    ?? null;
  const values = new Map<string, string>();
  if (description) {
    for (const child of Array.from(description.childNodes)) {
      if (child.nodeType === 1 && child.namespaceURI === XVEON_NS) {
        values.set(child.localName ?? '', child.textContent ?? '');
      }
    }
    for (const attr of Array.from(description.attributes)) {
      if (attr.namespaceURI === XVEON_NS) values.set(attr.localName ?? '', attr.value);
    }
  }
  return { doc, rdf, description, values };
}

function hasXveonProperty(element: Element): boolean {
  for (const attr of Array.from(element.attributes)) if (attr.namespaceURI === XVEON_NS) return true;
  return Array.from(element.childNodes).some((child) => child.nodeType === 1 && child.namespaceURI === XVEON_NS);
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
function knownName(name: string): boolean {
  return name === 'SchemaVersion' || name === 'Look' || name === 'Demosaic' || name === 'ModelSize' || name === 'ModelHash'
    || Object.hasOwn(PRE, name) || (name.startsWith('Odrt_') && DRT_KEYS.has(name.slice(5)));
}
function readValues(values: Map<string, string>): { edit: PhotoEdit; schemaVersion: number } {
  const edit = defaultPhotoEdit();
  const rawVersion = values.get('SchemaVersion');
  const schemaVersion = rawVersion === undefined ? 1 : numberValue('SchemaVersion', rawVersion);
  if (!Number.isInteger(schemaVersion) || schemaVersion < 1) throw new Error('SchemaVersion: invalid version');
  const look = values.get('Look');
  if (look !== undefined) {
    if (!LOOKS.has(look as LookPreset)) throw new Error('Look: invalid preset');
    edit.lookPreset = look as LookPreset;
  }
  const method = values.get('Demosaic');
  if (method !== undefined) {
    if (!METHODS.has(method as DemosaicMethod)) throw new Error('Demosaic: invalid method');
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
    if (!size || !SIZES.has(size as ModelSize)) throw new Error('ModelSize: invalid size');
    if (!hash) throw new Error('ModelHash: missing hash');
    edit.model = { size: size as ModelSize, sha256: hash };
  }
  return { edit, schemaVersion };
}

export function readSidecar(text: string | null): SidecarState {
  if (text === null) return { kind: 'absent' };
  try {
    const parsed = parse(text);
    if (![...parsed.values.keys()].some(knownName)) return { kind: 'absent' };
    const { edit, schemaVersion } = readValues(parsed.values);
    return schemaVersion > 1 ? { kind: 'newer', edit, schemaVersion } : { kind: 'ok', edit };
  } catch (error) {
    return { kind: 'unreadable', reason: error instanceof Error ? error.message : String(error) };
  }
}

function writeNumber(name: string, value: number): string {
  if (!Number.isFinite(value)) throw new Error(`${name}: non-finite number`);
  return String(value);
}
function attributesFor(edit: PhotoEdit): Map<string, string> {
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
  return result;
}

function emptyEnvelope(parsed: Parsed): boolean {
  const { doc, rdf } = parsed;
  const root = doc.documentElement;
  if (!root || root.localName !== 'xmpmeta' || root.namespaceURI !== 'adobe:ns:meta/') return false;
  const descriptions = Array.from(rdf.getElementsByTagNameNS(RDF_NS, 'Description'));
  if (descriptions.length !== 1 || descriptions[0].getAttributeNS(RDF_NS, 'about') !== '') return false;
  const onlyWhitespace = (node: Node) => Array.from(node.childNodes).every((child) => child.nodeType === 3 && !(child.nodeValue ?? '').trim());
  if (!onlyWhitespace(descriptions[0])) return false;
  if (Array.from(descriptions[0].attributes).some((a) => a.name !== 'rdf:about' && a.name !== 'xmlns:xveon')) return false;
  if (Array.from(root.attributes).some((a) => a.name !== 'xmlns:x')) return false;
  if (Array.from(rdf.attributes).some((a) => a.name !== 'xmlns:rdf')) return false;
  return Array.from(root.childNodes).every((child) => child === rdf || (child.nodeType === 3 && !(child.nodeValue ?? '').trim()))
    && Array.from(rdf.childNodes).every((child) => child === descriptions[0] || (child.nodeType === 3 && !(child.nodeValue ?? '').trim()))
    && Array.from(doc.childNodes).every((child) => child === root || (child.nodeType === 3 && !(child.nodeValue ?? '').trim()));
}

export function mergeSidecar(existing: string | null, edit: PhotoEdit): string | null {
  const attrs = attributesFor(edit);
  if (existing === null && attrs.size === 0) return null;
  const state = readSidecar(existing);
  if (state.kind === 'unreadable' || state.kind === 'newer') throw new Error(`Sidecar is ${state.kind}: ${state.kind === 'unreadable' ? state.reason : state.schemaVersion}`);
  if (existing === null) {
    const escaped = (value: string) => value.replaceAll('&', '&amp;').replaceAll('\"', '&quot;').replaceAll('<', '&lt;');
    const properties = [`xveon:SchemaVersion=\"1\"`, ...Array.from(attrs, ([name, value]) => `xveon:${name}=\"${escaped(value)}\"`)];
    return NEW_FILE.replace(`xmlns:xveon=\"${XVEON_NS}\"/>`, `xmlns:xveon=\"${XVEON_NS}\" ${properties.join(' ')}/>`);
  }
  const parsed = parse(existing);
  const description = parsed.description ?? parsed.doc.createElementNS(RDF_NS, 'rdf:Description');
  if (!parsed.description) {
    description.setAttributeNS(RDF_NS, 'rdf:about', '');
    parsed.rdf.appendChild(description);
  }
  for (const attr of Array.from(description.attributes)) {
    if (attr.namespaceURI === XVEON_NS && knownName(attr.localName ?? '')) description.removeAttributeNode(attr);
  }
  for (const child of Array.from(description.childNodes)) {
    if (child.nodeType === 1 && child.namespaceURI === XVEON_NS && knownName(child.localName ?? '')) description.removeChild(child);
  }
  const hasUnknown = hasXveonProperty(description);
  if (attrs.size > 0 || hasUnknown) {
    let prefix = 'xveon';
    for (let suffix = 1; description.lookupNamespaceURI(prefix) !== null && description.lookupNamespaceURI(prefix) !== XVEON_NS; suffix++) {
      prefix = `xveon${suffix}`;
    }
    if (description.lookupNamespaceURI(prefix) !== XVEON_NS) description.setAttributeNS(XMLNS_NS, `xmlns:${prefix}`, XVEON_NS);
    description.setAttributeNS(XVEON_NS, `${prefix}:SchemaVersion`, '1');
    for (const [name, value] of attrs) description.setAttributeNS(XVEON_NS, `${prefix}:${name}`, value);
  } else if (description.getAttributeNS(XMLNS_NS, 'xveon') === XVEON_NS) {
    description.removeAttributeNS(XMLNS_NS, 'xveon');
  }
  if (emptyEnvelope(parsed)) return null;
  const result = new XMLSerializer().serializeToString(parsed.doc);
  return existing.startsWith('\uFEFF') ? `\uFEFF${result}` : result;
}
