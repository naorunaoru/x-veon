import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { DOMParser, XMLSerializer } from '@xmldom/xmldom';
import { describe, expect, it } from 'vitest';
import { defaultPhotoEdit } from '@/app/photo-edit';
import { DEFAULT_LENS_EDIT } from '@/lib/lens/edit';
import { mergeSidecar, readSidecar, XVEON_NS } from './xmp';

const RDF_NS = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#';
const XMP_NS = 'http://ns.adobe.com/xap/1.0/';
const fixture = (name: string) => readFileSync(fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url)), 'utf8');
function withoutOurs(xml: string): string {
  const doc = new DOMParser().parseFromString(xml.replace(/^\uFEFF/, ''), 'application/xml');
  const nodes = Array.from(doc.getElementsByTagName('*'));
  for (const node of nodes) {
    for (let i = node.attributes.length - 1; i >= 0; i--) {
      const a = node.attributes.item(i)!;
      if (a.namespaceURI === XVEON_NS || a.name === 'xmlns:xveon' || (a.namespaceURI === XMP_NS && a.localName === 'Rating')) node.removeAttributeNode(a);
    }
    if (node.namespaceURI === XVEON_NS || (node.namespaceURI === XMP_NS && node.localName === 'Rating')) node.parentNode?.removeChild(node);
  }
  return new XMLSerializer().serializeToString(doc);
}
function descriptions(xml: string) {
  return Array.from(new DOMParser().parseFromString(xml.replace(/^\uFEFF/, ''), 'application/xml').getElementsByTagNameNS(RDF_NS, 'Description'));
}

describe('XMP sidecar codec', () => {
  it('keeps everything that is not ours', () => {
    const before = fixture('darktable-sony_a6400_21.arw.xmp');
    const after = mergeSidecar(before, { ...defaultPhotoEdit(), preProcessOverrides: { exposure: 0.3 } }, 0)!;
    expect(withoutOurs(after)).toBe(withoutOurs(before));
    expect(readSidecar(after)).toEqual({ kind: 'ok', rating: 0, edit: expect.objectContaining({ preProcessOverrides: { exposure: 0.3 } }) });
  });

  it('does not create a file for an empty edit', () => {
    expect(mergeSidecar(null, defaultPhotoEdit(), 0)).toBeNull();
  });

  it('deletes an envelope with only known properties on reset', () => {
    const xml = mergeSidecar(null, { ...defaultPhotoEdit(), lookPreset: 'umbra' }, 0)!;
    expect(mergeSidecar(xml, defaultPhotoEdit(), 0)).toBeNull();
  });

  it('keeps unknown X-Veon properties and schema version on reset', () => {
    const result = mergeSidecar(fixture('xveon-unknown-props.xmp'), defaultPhotoEdit(), 0)!;
    expect(result).toContain('xveon:FutureValue="keep"');
    expect(result).toContain('xveon:FutureElement');
    expect(result).toContain('xveon:SchemaVersion="1"');
    expect(result).not.toContain('xveon:Look=');
  });

  it('reads element properties and replaces them with attributes', () => {
    const before = fixture('xveon-element-form.xmp');
    expect(readSidecar(before)).toEqual({ kind: 'ok', rating: 0, edit: { ...defaultPhotoEdit(), preProcessOverrides: { exposure: 0.3 }, openDrtOverrides: { tn_lcon_enable: true } } });
    const after = mergeSidecar(before, { ...defaultPhotoEdit(), preProcessOverrides: { exposure: 0.4 } }, 0)!;
    expect(after).toContain('xveon:Exposure="0.4"');
    expect(after).not.toContain('<xveon:Exposure>');
  });

  it('reads BOM and CRLF and preserves foreign content on write', () => {
    const before = fixture('bom-crlf.xmp');
    expect(readSidecar(before)).toEqual({ kind: 'absent' });
    const after = mergeSidecar(before, { ...defaultPhotoEdit(), lookPreset: 'colorful' }, 0)!;
    expect(withoutOurs(after)).toBe(withoutOurs(before));
    expect(after).toContain('foreign:Note="keep"');
  });

  it('uses only the first empty-about description when none holds our properties', () => {
    const before = fixture('multi-description.xmp');
    const after = mergeSidecar(before, { ...defaultPhotoEdit(), lookPreset: 'umbra' }, 0)!;
    const descriptionsAfter = descriptions(after);
    expect(descriptionsAfter).toHaveLength(3);
    expect(descriptionsAfter[0].getAttributeNS(XVEON_NS, 'Look')).toBeNull();
    expect(descriptionsAfter[1].getAttributeNS(XVEON_NS, 'Look')).toBe('umbra');
    expect(descriptionsAfter[2].getAttributeNS(XVEON_NS, 'Look')).toBeNull();
    expect(withoutOurs(after)).toBe(withoutOurs(before));
  });

  it('uses the first description already carrying our properties', () => {
    const source = '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about=""/><rdf:Description rdf:about="urn:other" xmlns:xveon="https://naorunaoru.github.io/x-veon/ns/1.0/" xveon:Look="umbra"/></rdf:RDF></x:xmpmeta>';
    const result = descriptions(mergeSidecar(source, { ...defaultPhotoEdit(), lookPreset: 'colorful' }, 0)!);
    expect(result[0].getAttributeNS(XVEON_NS, 'Look')).toBeNull();
    expect(result[1].getAttributeNS(XVEON_NS, 'Look')).toBe('colorful');
  });

  it('treats a newer schema as view-only and rejects writes', () => {
    const source = fixture('xveon-schema-2.xmp');
    expect(readSidecar(source)).toEqual({ kind: 'newer', rating: 0, schemaVersion: 2, edit: { ...defaultPhotoEdit(), preProcessOverrides: { exposure: 0.3 } } });
    expect(() => mergeSidecar(source, defaultPhotoEdit(), 0)).toThrow();
  });

  it('treats malformed known properties as unreadable and rejects writes', () => {
    const source = fixture('malformed.xmp');
    expect(readSidecar(source)).toEqual({ kind: 'unreadable', reason: expect.stringContaining('Exposure') });
    expect(() => mergeSidecar(source, defaultPhotoEdit(), 0)).toThrow();
  });

  it('keeps Apple, Lightroom and digiKam foreign properties', () => {
    const apple = fixture('apple-DSCF8199.xmp');
    expect(readSidecar(apple)).toEqual({ kind: 'absent' });
    for (const [name, marker] of [['apple-DSCF8199.xmp', 'photoshop:DateCreated'], ['lightroom-synthetic.xmp', 'crs:Exposure2012'], ['digikam-synthetic.xmp', 'digiKam:Rating']]) {
      const before = fixture(name);
      const after = mergeSidecar(before, { ...defaultPhotoEdit(), lookPreset: 'umbra' }, 0)!;
      expect(after).toContain(marker);
      expect(withoutOurs(after)).toBe(withoutOurs(before));
    }
  });

  it('appends a new description when every existing rdf:about is nonempty', () => {
    const before = '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="urn:other"/></rdf:RDF></x:xmpmeta>';
    const after = mergeSidecar(before, { ...defaultPhotoEdit(), lookPreset: 'umbra' }, 0)!;
    const items = descriptions(after);
    expect(items).toHaveLength(2);
    expect(items[0].getAttributeNS(XVEON_NS, 'Look')).toBeNull();
    expect(items[1].getAttributeNS(RDF_NS, 'about')).toBe('');
    expect(items[1].getAttributeNS(XVEON_NS, 'Look')).toBe('umbra');
  });

  it('keeps unknown X-Veon properties named like Object prototype members', () => {
    const before = `<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="${RDF_NS}"><rdf:Description rdf:about="" xmlns:xveon="${XVEON_NS}" xveon:SchemaVersion="1" xveon:constructor="foreign"><xveon:toString>foreign element</xveon:toString></rdf:Description></rdf:RDF></x:xmpmeta>`;
    const after = mergeSidecar(before, defaultPhotoEdit(), 0)!;
    expect(after).toContain('xveon:constructor="foreign"');
    expect(after).toContain('<xveon:toString>foreign element</xveon:toString>');
    expect(after).toContain('xveon:SchemaVersion="1"');
  });

  it('preserves a foreign xveon prefix binding while writing our namespace', () => {
    const before = `<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="${RDF_NS}"><rdf:Description rdf:about="" xmlns:xveon="urn:foreign" xmlns:ours="${XVEON_NS}" xveon:Foreign="keep" ours:Future="unknown" ours:SchemaVersion="1"/></rdf:RDF></x:xmpmeta>`;
    const after = mergeSidecar(before, { ...defaultPhotoEdit(), lookPreset: 'umbra' }, 0)!;
    const errors: string[] = [];
    const doc = new DOMParser({ onError: (_level, message) => { errors.push(message); } }).parseFromString(after, 'application/xml');
    expect(errors).toEqual([]);
    const item = doc.getElementsByTagNameNS(RDF_NS, 'Description').item(0)!;
    expect(item.getAttribute('xmlns:xveon')).toBe('urn:foreign');
    expect(item.getAttributeNS('urn:foreign', 'Foreign')).toBe('keep');
    expect(item.getAttributeNS(XVEON_NS, 'Look')).toBe('umbra');
    expect(readSidecar(after)).toEqual({ kind: 'ok', rating: 0, edit: { ...defaultPhotoEdit(), lookPreset: 'umbra' } });
    const reset = mergeSidecar(after, defaultPhotoEdit(), 0)!;
    const resetItem = descriptions(reset)[0];
    expect(resetItem.getAttribute('xmlns:xveon')).toBe('urn:foreign');
    expect(resetItem.getAttributeNS('urn:foreign', 'Foreign')).toBe('keep');
    expect(resetItem.hasAttributeNS(XVEON_NS, 'Look')).toBe(false);
    expect(resetItem.getAttributeNS(XVEON_NS, 'Future')).toBe('unknown');
    expect(resetItem.getAttributeNS(XVEON_NS, 'SchemaVersion')).toBe('1');
  });

  it('round-trips a lens block as flat attributes and writes nothing for the defaults', () => {
    const lens = { ...DEFAULT_LENS_EDIT, vignetting: false, source: { lensfun: { file: 'misc.json', model: '23mmF1.4XM' } }, manual: { ...DEFAULT_LENS_EDIT.manual, caRed: 1.5 } };
    const xml = mergeSidecar(null, { ...defaultPhotoEdit(), lens }, 0)!;
    expect(xml).toContain('xveon:LensSource="lensfun"');
    expect(xml).toContain('xveon:LensModel="23mmF1.4XM"');
    expect(xml).toContain('xveon:LensVignetting="False"');
    expect(xml).toContain('xveon:LensManualCaRed="1.5"');
    expect(xml).not.toContain('LensDistortion');
    expect(xml).toContain('xveon:SchemaVersion="1"');
    expect(readSidecar(xml)).toEqual({ kind: 'ok', rating: 0, edit: expect.objectContaining({ lens }) });
    expect(mergeSidecar(null, { ...defaultPhotoEdit(), lens: { ...DEFAULT_LENS_EDIT } }, 0)).toBeNull();
  });

  it('removes the lens attributes when the block returns to the defaults', () => {
    const xml = mergeSidecar(null, { ...defaultPhotoEdit(), lens: { ...DEFAULT_LENS_EDIT, ca: false, source: 'manual' } }, 0)!;
    expect(xml).toContain('xveon:LensCa="False"');
    expect(mergeSidecar(xml, defaultPhotoEdit(), 0)).toBeNull();
  });

  it('writes xmp:Rating as an attribute and removes it at 0', () => {
    const xml = mergeSidecar(null, defaultPhotoEdit(), 4)!;
    expect(xml).toContain('xmp:Rating="4"');
    expect(xml).toContain(`xmlns:xmp="${XMP_NS}"`);
    expect(xml).not.toContain('xveon:SchemaVersion');
    expect(readSidecar(xml)).toEqual({ kind: 'ok', edit: defaultPhotoEdit(), rating: 4 });
    expect(mergeSidecar(xml, defaultPhotoEdit(), 0)).toBeNull();
  });

  it('reads a foreign rating without our properties', () => {
    expect(readSidecar(fixture('lightroom-rated.xmp'))).toEqual({ kind: 'ok', rating: 3, edit: defaultPhotoEdit() });
  });

  it('removes only xmp:Rating on clear', () => {
    const after = mergeSidecar(fixture('lightroom-rated.xmp'), defaultPhotoEdit(), 0);
    expect(after).not.toBeNull();
    expect(after).toContain('xmp:Label="Red"');
    expect(after).toContain(`xmlns:xmp="${XMP_NS}"`);
    expect(after).not.toContain('xmp:Rating');
  });

  it('reads xmp:Rating in element form and rewrites it as an attribute', () => {
    const before = `<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="${RDF_NS}"><rdf:Description rdf:about="" xmlns:xmp="${XMP_NS}"><xmp:Rating>2</xmp:Rating></rdf:Description></rdf:RDF></x:xmpmeta>`;
    expect(readSidecar(before)).toEqual({ kind: 'ok', rating: 2, edit: defaultPhotoEdit() });
    const item = descriptions(mergeSidecar(before, defaultPhotoEdit(), 5)!)[0];
    expect(item.getElementsByTagNameNS(XMP_NS, 'Rating')).toHaveLength(0);
    expect(item.getAttributeNS(XMP_NS, 'Rating')).toBe('5');
  });

  it('reuses a foreign prefix bound to the xmp namespace', () => {
    const before = `<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="${RDF_NS}"><rdf:Description rdf:about="" xmlns:x1="${XMP_NS}" x1:Label="Red"/></rdf:RDF></x:xmpmeta>`;
    const after = mergeSidecar(before, defaultPhotoEdit(), 1)!;
    expect(after).toContain('x1:Rating="1"');
    expect(after.split(XMP_NS)).toHaveLength(2);
    expect(readSidecar(after)).toEqual({ kind: 'ok', rating: 1, edit: defaultPhotoEdit() });
  });

  it('treats a malformed rating as unreadable', () => {
    const source = `<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="${RDF_NS}"><rdf:Description rdf:about="" xmlns:xmp="${XMP_NS}" xmlns:xveon="${XVEON_NS}" xmp:Rating="banana" xveon:Look="umbra"/></rdf:RDF></x:xmpmeta>`;
    expect(readSidecar(source)).toEqual({ kind: 'unreadable', reason: expect.stringMatching(/xmp:Rating: invalid rating/) });
  });

  it('drops a foreign rating of 0 and keeps the rest of a darktable file', () => {
    const before = fixture('darktable-sony_a6400_21.arw.xmp');
    const after = mergeSidecar(before, { ...defaultPhotoEdit(), lookPreset: 'umbra' }, 0)!;
    const item = descriptions(after)[0];
    expect(item.hasAttributeNS(XMP_NS, 'Rating')).toBe(false);
    expect(item.getAttributeNS(XVEON_NS, 'Look')).toBe('umbra');
    const darktable = (xml: string) => Array.from(descriptions(xml)[0].attributes).filter((a) => a.name.startsWith('darktable:')).map((a) => `${a.name}=${a.value}`);
    expect(darktable(before).length).toBeGreaterThan(0);
    expect(darktable(after)).toEqual(darktable(before));
    expect(withoutOurs(after)).toBe(withoutOurs(before));
  });
});
