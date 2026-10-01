import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { DOMParser, XMLSerializer } from '@xmldom/xmldom';
import { describe, expect, it } from 'vitest';
import { defaultPhotoEdit } from '@/app/photo-edit';
import { mergeSidecar, readSidecar, XVEON_NS } from './xmp';

const RDF_NS = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#';
const fixture = (name: string) => readFileSync(fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url)), 'utf8');
function withoutXveon(xml: string): string {
  const doc = new DOMParser().parseFromString(xml.replace(/^\uFEFF/, ''), 'application/xml');
  const nodes = Array.from(doc.getElementsByTagName('*'));
  for (const node of nodes) {
    for (let i = node.attributes.length - 1; i >= 0; i--) {
      const a = node.attributes.item(i)!;
      if (a.namespaceURI === XVEON_NS || a.name === 'xmlns:xveon') node.removeAttributeNode(a);
    }
    if (node.namespaceURI === XVEON_NS) node.parentNode?.removeChild(node);
  }
  return new XMLSerializer().serializeToString(doc);
}
function descriptions(xml: string) {
  return Array.from(new DOMParser().parseFromString(xml.replace(/^\uFEFF/, ''), 'application/xml').getElementsByTagNameNS(RDF_NS, 'Description'));
}

describe('XMP sidecar codec', () => {
  it('keeps everything that is not ours', () => {
    const before = fixture('darktable-sony_a6400_21.arw.xmp');
    const after = mergeSidecar(before, { ...defaultPhotoEdit(), preProcessOverrides: { exposure: 0.3 } })!;
    expect(withoutXveon(after)).toBe(withoutXveon(before));
    expect(readSidecar(after)).toEqual({ kind: 'ok', edit: expect.objectContaining({ preProcessOverrides: { exposure: 0.3 } }) });
  });

  it('round-trips a full edit, including a neural model and boolean', () => {
    const edit = { ...defaultPhotoEdit(), lookPreset: 'umbra' as const, demosaicMethod: 'neural-net' as const,
      model: { size: 'M' as const, sha256: 'abc123' }, preProcessOverrides: { exposure: 0.1, wb_temp: -2.4, wb_tint: 1e-7, sharpen_amount: 250 },
      openDrtOverrides: { tn_con: 1.2, tn_lcon_enable: false } };
    const xml = mergeSidecar(null, edit)!;
    expect(xml).toContain('xveon:Odrt_tn_lcon_enable="False"');
    expect(xml).toContain('xveon:ModelSize="M"');
    expect(readSidecar(xml)).toEqual({ kind: 'ok', edit });
  });

  it('does not create a file for an empty edit', () => {
    expect(mergeSidecar(null, defaultPhotoEdit())).toBeNull();
  });

  it('deletes an envelope with only known properties on reset', () => {
    const xml = mergeSidecar(null, { ...defaultPhotoEdit(), lookPreset: 'umbra' })!;
    expect(mergeSidecar(xml, defaultPhotoEdit())).toBeNull();
  });

  it('keeps unknown X-Veon properties and schema version on reset', () => {
    const result = mergeSidecar(fixture('xveon-unknown-props.xmp'), defaultPhotoEdit())!;
    expect(result).toContain('xveon:FutureValue="keep"');
    expect(result).toContain('xveon:FutureElement');
    expect(result).toContain('xveon:SchemaVersion="1"');
    expect(result).not.toContain('xveon:Look=');
  });

  it('reads element properties and replaces them with attributes', () => {
    const before = fixture('xveon-element-form.xmp');
    expect(readSidecar(before)).toEqual({ kind: 'ok', edit: { ...defaultPhotoEdit(), preProcessOverrides: { exposure: 0.3 }, openDrtOverrides: { tn_lcon_enable: true } } });
    const after = mergeSidecar(before, { ...defaultPhotoEdit(), preProcessOverrides: { exposure: 0.4 } })!;
    expect(after).toContain('xveon:Exposure="0.4"');
    expect(after).not.toContain('<xveon:Exposure>');
  });

  it('reads BOM and CRLF and preserves foreign content on write', () => {
    const before = fixture('bom-crlf.xmp');
    expect(readSidecar(before)).toEqual({ kind: 'absent' });
    const after = mergeSidecar(before, { ...defaultPhotoEdit(), lookPreset: 'colorful' })!;
    expect(withoutXveon(after)).toBe(withoutXveon(before));
    expect(after).toContain('foreign:Note="keep"');
  });

  it('uses only the first empty-about description when none holds our properties', () => {
    const before = fixture('multi-description.xmp');
    const after = mergeSidecar(before, { ...defaultPhotoEdit(), lookPreset: 'umbra' })!;
    const descriptionsAfter = descriptions(after);
    expect(descriptionsAfter).toHaveLength(3);
    expect(descriptionsAfter[0].getAttributeNS(XVEON_NS, 'Look')).toBeNull();
    expect(descriptionsAfter[1].getAttributeNS(XVEON_NS, 'Look')).toBe('umbra');
    expect(descriptionsAfter[2].getAttributeNS(XVEON_NS, 'Look')).toBeNull();
    expect(withoutXveon(after)).toBe(withoutXveon(before));
  });

  it('uses the first description already carrying our properties', () => {
    const source = '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about=""/><rdf:Description rdf:about="urn:other" xmlns:xveon="https://naorunaoru.github.io/x-veon/ns/1.0/" xveon:Look="umbra"/></rdf:RDF></x:xmpmeta>';
    const result = descriptions(mergeSidecar(source, { ...defaultPhotoEdit(), lookPreset: 'colorful' })!);
    expect(result[0].getAttributeNS(XVEON_NS, 'Look')).toBeNull();
    expect(result[1].getAttributeNS(XVEON_NS, 'Look')).toBe('colorful');
  });

  it('treats a newer schema as view-only and rejects writes', () => {
    const source = fixture('xveon-schema-2.xmp');
    expect(readSidecar(source)).toEqual({ kind: 'newer', schemaVersion: 2, edit: { ...defaultPhotoEdit(), preProcessOverrides: { exposure: 0.3 } } });
    expect(() => mergeSidecar(source, defaultPhotoEdit())).toThrow();
  });

  it('treats malformed known properties as unreadable and rejects writes', () => {
    const source = fixture('malformed.xmp');
    expect(readSidecar(source)).toEqual({ kind: 'unreadable', reason: expect.stringContaining('Exposure') });
    expect(() => mergeSidecar(source, defaultPhotoEdit())).toThrow();
  });

  it('formats numeric overrides with String(number) and rejects non-finite values', () => {
    const xml = mergeSidecar(null, { ...defaultPhotoEdit(), preProcessOverrides: { exposure: 0.1, wb_temp: -2.4, wb_tint: 1e-7, sharpen_amount: 250 } })!;
    expect(xml).toContain('xveon:Exposure="0.1"');
    expect(xml).toContain('xveon:WbTemp="-2.4"');
    expect(xml).toContain('xveon:WbTint="1e-7"');
    expect(xml).toContain('xveon:Sharpen="250"');
    expect(() => mergeSidecar(null, { ...defaultPhotoEdit(), preProcessOverrides: { exposure: Infinity } })).toThrow();
  });

  it('keeps Apple, Lightroom and digiKam foreign properties', () => {
    const apple = fixture('apple-DSCF8199.xmp');
    expect(readSidecar(apple)).toEqual({ kind: 'absent' });
    for (const [name, marker] of [['apple-DSCF8199.xmp', 'photoshop:DateCreated'], ['lightroom-synthetic.xmp', 'crs:Exposure2012'], ['digikam-synthetic.xmp', 'digiKam:Rating']]) {
      const before = fixture(name);
      const after = mergeSidecar(before, { ...defaultPhotoEdit(), lookPreset: 'umbra' })!;
      expect(after).toContain(marker);
      expect(withoutXveon(after)).toBe(withoutXveon(before));
    }
  });
});
