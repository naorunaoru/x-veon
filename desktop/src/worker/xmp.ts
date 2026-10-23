import { DOMParser, XMLSerializer } from '@xmldom/xmldom';
import type { Document, Element, Node } from '@xmldom/xmldom';
import { decodeSidecar, encodeSidecar, isKnownProperty, unknownProperties, XMP_NS, XVEON_NS } from '@/app/sidecar';
import type { SidecarProperties } from '@/app/sidecar';
import type { PhotoEdit, Rating } from '@/host';

export { XVEON_NS } from '@/app/sidecar';
const RDF_NS = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#';
const XMLNS_NS = 'http://www.w3.org/2000/xmlns/';
const NEW_FILE = `<x:xmpmeta xmlns:x="adobe:ns:meta/">\n <rdf:RDF xmlns:rdf="${RDF_NS}">\n  <rdf:Description rdf:about="" xmlns:xveon="${XVEON_NS}"/>\n </rdf:RDF>\n</x:xmpmeta>`;

export type SidecarState =
  | { kind: 'absent' }
  | { kind: 'ok'; edit: PhotoEdit; rating: Rating }
  | { kind: 'newer'; edit: PhotoEdit; rating: Rating; schemaVersion: number }
  | { kind: 'unreadable'; reason: string };

type Parsed = { doc: Document; rdf: Element; description: Element | null; values: SidecarProperties };

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
  const values: SidecarProperties = {};
  if (description) {
    for (const child of Array.from(description.childNodes)) {
      const name = child.nodeType === 1 ? propertyName(child) : null;
      if (name) values[name] = child.textContent ?? '';
    }
    for (const attr of Array.from(description.attributes)) {
      const name = propertyName(attr);
      if (name) values[name] = attr.value;
    }
  }
  return { doc, rdf, description, values };
}

function hasXveonProperty(element: Element): boolean {
  for (const attr of Array.from(element.attributes)) if (attr.namespaceURI === XVEON_NS) return true;
  return Array.from(element.childNodes).some((child) => child.nodeType === 1 && child.namespaceURI === XVEON_NS);
}

/** 'xveon:<local>' for our namespace, 'xmp:Rating' for the XMP rating, null for anything else. */
function propertyName(node: Node): string | null {
  if (node.namespaceURI === XVEON_NS) return `xveon:${node.localName ?? ''}`;
  return node.namespaceURI === XMP_NS && node.localName === 'Rating' ? 'xmp:Rating' : null;
}

export function readSidecar(text: string | null): SidecarState {
  if (text === null) return { kind: 'absent' };
  try {
    const parsed = parse(text);
    if (!Object.keys(parsed.values).some(isKnownProperty)) return { kind: 'absent' };
    const { edit, rating, schemaVersion } = decodeSidecar(parsed.values);
    return schemaVersion > 1 ? { kind: 'newer', edit, rating, schemaVersion } : { kind: 'ok', edit, rating };
  } catch (error) {
    return { kind: 'unreadable', reason: error instanceof Error ? error.message : String(error) };
  }
}

function boundPrefix(element: Element, namespace: string, preferred: string): string {
  let prefix = preferred;
  for (let suffix = 1; element.lookupNamespaceURI(prefix) !== null && element.lookupNamespaceURI(prefix) !== namespace; suffix++) {
    prefix = `${preferred}${suffix}`;
  }
  if (element.lookupNamespaceURI(prefix) !== namespace) element.setAttributeNS(XMLNS_NS, `xmlns:${prefix}`, namespace);
  return prefix;
}

function emptyEnvelope(parsed: Parsed): boolean {
  const { doc, rdf } = parsed;
  const root = doc.documentElement;
  if (!root || root.localName !== 'xmpmeta' || root.namespaceURI !== 'adobe:ns:meta/') return false;
  const descriptions = Array.from(rdf.getElementsByTagNameNS(RDF_NS, 'Description'));
  if (descriptions.length !== 1 || descriptions[0].getAttributeNS(RDF_NS, 'about') !== '') return false;
  const onlyWhitespace = (node: Node) => Array.from(node.childNodes).every((child) => child.nodeType === 3 && !(child.nodeValue ?? '').trim());
  if (!onlyWhitespace(descriptions[0])) return false;
  if (Array.from(descriptions[0].attributes).some((a) => a.name !== 'rdf:about' && a.name !== 'xmlns:xveon' && a.name !== 'xmlns:xmp')) return false;
  if (Array.from(root.attributes).some((a) => a.name !== 'xmlns:x')) return false;
  if (Array.from(rdf.attributes).some((a) => a.name !== 'xmlns:rdf')) return false;
  return Array.from(root.childNodes).every((child) => child === rdf || (child.nodeType === 3 && !(child.nodeValue ?? '').trim()))
    && Array.from(rdf.childNodes).every((child) => child === descriptions[0] || (child.nodeType === 3 && !(child.nodeValue ?? '').trim()))
    && Array.from(doc.childNodes).every((child) => child === root || (child.nodeType === 3 && !(child.nodeValue ?? '').trim()));
}

export function mergeSidecar(existing: string | null, edit: PhotoEdit, rating: Rating): string | null {
  const state = readSidecar(existing);
  if (state.kind === 'unreadable' || state.kind === 'newer') throw new Error(`Sidecar is ${state.kind}: ${state.kind === 'unreadable' ? state.reason : state.schemaVersion}`);
  const parsed = parse(existing ?? NEW_FILE);
  const encoded = Object.entries(encodeSidecar({ edit, rating }, unknownProperties(parsed.values))).filter(([name]) => isKnownProperty(name));
  const description = parsed.description ?? parsed.doc.createElementNS(RDF_NS, 'rdf:Description');
  if (!parsed.description) {
    description.setAttributeNS(RDF_NS, 'rdf:about', '');
    parsed.rdf.appendChild(description);
  }
  const known = (node: Node) => { const name = propertyName(node); return name !== null && isKnownProperty(name); };
  for (const attr of Array.from(description.attributes)) if (known(attr)) description.removeAttributeNode(attr);
  for (const child of Array.from(description.childNodes)) if (child.nodeType === 1 && known(child)) description.removeChild(child);
  const xveon = encoded.filter(([name]) => name.startsWith('xveon:'));
  if (xveon.length > 0) {
    const prefix = boundPrefix(description, XVEON_NS, 'xveon');
    for (const [name, value] of xveon) description.setAttributeNS(XVEON_NS, `${prefix}:${name.slice('xveon:'.length)}`, value);
  } else if (description.getAttributeNS(XMLNS_NS, 'xveon') === XVEON_NS) {
    description.removeAttributeNS(XMLNS_NS, 'xveon');
  }
  const ratingValue = encoded.find(([name]) => name === 'xmp:Rating')?.[1];
  if (ratingValue !== undefined) {
    const prefix = description.lookupPrefix(XMP_NS) || boundPrefix(description, XMP_NS, 'xmp');
    description.setAttributeNS(XMP_NS, `${prefix}:Rating`, ratingValue);
  }
  if (emptyEnvelope(parsed)) return null;
  const result = new XMLSerializer().serializeToString(parsed.doc);
  return existing?.startsWith('\uFEFF') ? `\uFEFF${result}` : result;
}
