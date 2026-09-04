/**
 * The one-way dependency rule between source layers (web-consolidation design, §4).
 * Uses the TypeScript parser so every import form is covered and violations include file:line.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname, resolve, relative, sep } from 'node:path';
import ts from 'typescript';

// Vitest serves modules through Vite, so import.meta.url is not guaranteed to be a file URL.
const SRC = resolve(process.cwd(), 'src');

type Layer = 'lib' | 'gpu' | 'pipeline' | 'renderer' | 'app' | 'components' | 'dev' | 'root';
const LAYER_DIRS = ['lib', 'gpu', 'pipeline', 'renderer', 'app', 'components', 'dev'] as const;

const ALLOWED: Record<Layer, readonly Layer[]> = {
  lib: ['lib'],
  gpu: ['gpu', 'lib'],
  pipeline: ['pipeline', 'gpu', 'lib'],
  renderer: ['renderer', 'gpu', 'lib'],
  app: ['app', 'pipeline', 'renderer', 'gpu', 'lib'],
  components: ['components', 'app', 'renderer', 'lib'],
  dev: ['dev', 'components', 'app', 'pipeline', 'renderer', 'gpu', 'lib', 'root'],
  root: ['root', 'components', 'app', 'renderer', 'lib', 'dev'],
};

const UI_PACKAGES = /^(react|react-dom|zustand|lucide-react)(\/|$)|^@radix-ui\//;
const FRAMEWORK_FREE: readonly Layer[] = ['lib', 'gpu', 'pipeline', 'renderer'];

/** Layer of a file or directory under src: its first path segment, or `root` for src itself. */
function layerOf(absPath: string): Layer {
  const top = relative(SRC, absPath).split(sep)[0];
  return (LAYER_DIRS as readonly string[]).includes(top) ? (top as Layer) : 'root';
}

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) {
      if (name !== 'test') walk(path, out);
      continue;
    }
    if (!/\.(ts|tsx)$/.test(name) || /\.d\.ts$/.test(name) || /\.test\.tsx?$/.test(name)) continue;
    out.push(path);
  }
  return out;
}

interface Ref { spec: string; line: number }

/** Every module specifier in `source`: static and dynamic imports, re-exports, and `new URL(...)`. */
function refsOf(file: string, source: string): Ref[] {
  const sourceFile = ts.createSourceFile(
    file,
    source,
    ts.ScriptTarget.Latest,
    true,
    file.endsWith('x') ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );
  const refs: Ref[] = [];
  const add = (node: ts.Node, spec: string) => refs.push({
    spec,
    line: sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1,
  });
  const visit = (node: ts.Node): void => {
    if (
      (ts.isImportDeclaration(node) || ts.isExportDeclaration(node))
      && node.moduleSpecifier
      && ts.isStringLiteral(node.moduleSpecifier)
    ) {
      add(node, node.moduleSpecifier.text);
    } else if (
      ts.isCallExpression(node)
      && node.expression.kind === ts.SyntaxKind.ImportKeyword
      && node.arguments.length
      && ts.isStringLiteral(node.arguments[0])
    ) {
      add(node, node.arguments[0].text);
    } else if (
      ts.isNewExpression(node)
      && ts.isIdentifier(node.expression)
      && node.expression.text === 'URL'
      && node.arguments?.length
      && ts.isStringLiteral(node.arguments[0])
    ) {
      add(node, node.arguments[0].text);
    }
    ts.forEachChild(node, visit);
  };
  visit(sourceFile);
  return refs;
}

type Target = { kind: 'layer'; layer: Layer } | { kind: 'package'; name: string } | { kind: 'asset' };

function resolveTarget(file: string, spec: string): Target {
  const clean = spec.split('?')[0];
  if (/\.(css|wgsl|json)$/.test(clean)) return { kind: 'asset' };
  if (spec.startsWith('@/')) return { kind: 'layer', layer: layerOf(join(SRC, clean.slice(2))) };
  if (spec.startsWith('.')) {
    const absolute = resolve(dirname(file), clean);
    if (!absolute.startsWith(SRC + sep)) return { kind: 'asset' };
    return { kind: 'layer', layer: layerOf(absolute) };
  }
  return { kind: 'package', name: spec };
}

interface SourceText { file: string; source: string }

/** Rule violations for the given sources, each as `path:line (from) → spec [target]`. */
function violationsIn(files: SourceText[]): string[] {
  const violations: string[] = [];
  for (const { file, source } of files) {
    const from = layerOf(file);
    for (const { spec, line } of refsOf(file, source)) {
      const target = resolveTarget(file, spec);
      const where = `${relative(SRC, file)}:${line} (${from}) → ${spec}`;
      if (target.kind === 'layer' && !ALLOWED[from].includes(target.layer)) {
        violations.push(`${where} [${target.layer}]`);
      }
      if (target.kind === 'package' && FRAMEWORK_FREE.includes(from) && UI_PACKAGES.test(target.name)) {
        violations.push(`${where} [package]`);
      }
    }
  }
  return violations;
}

/** A file that exists only for the test, addressed by its would-be path under src. */
const fixture = (path: string, source: string): SourceText => ({ file: join(SRC, path), source });

describe('layer dependency rule', () => {
  const files = walk(SRC).map((file) => ({ file, source: readFileSync(file, 'utf8') }));

  it('sees the source tree', () => {
    expect(files.length).toBeGreaterThan(50);
  });

  it('has no violations', () => {
    expect(violationsIn(files)).toEqual([]);
  });

  it('flags every import form that crosses a layer boundary', () => {
    expect(violationsIn([
      fixture('lib/fixture.ts', "import '@/app/store';"),
      fixture('pipeline/decode/fixture.ts', "export const load = () => import('../../renderer/renderer');"),
      fixture('renderer/fixture.ts', "export { useAppStore } from '@/app/store';"),
      fixture('gpu/fixture.ts', "export const w = new Worker(new URL('../pipeline/demosaic/demosaic-worker.ts', import.meta.url));"),
      fixture('components/Fixture.tsx', "import { encodeImage } from '@/pipeline/export/encoder';"),
    ])).toEqual([
      'lib/fixture.ts:1 (lib) → @/app/store [app]',
      'pipeline/decode/fixture.ts:1 (pipeline) → ../../renderer/renderer [renderer]',
      'renderer/fixture.ts:1 (renderer) → @/app/store [app]',
      'gpu/fixture.ts:1 (gpu) → ../pipeline/demosaic/demosaic-worker.ts [pipeline]',
      'components/Fixture.tsx:1 (components) → @/pipeline/export/encoder [pipeline]',
    ]);
  });

  it('treats a bare directory import as that layer', () => {
    expect(violationsIn([
      fixture('components/Fixture.tsx', "import { processRaw } from '@/pipeline';"),
      fixture('Fixture.tsx', "import { processRaw } from '@/pipeline';"),
      fixture('app/fixture.ts', "import { processRaw } from '@/pipeline';"),
      fixture('app/hooks/fixture.ts', "import { processRaw } from '..';\nimport { processRaw as again } from '../../pipeline';"),
    ])).toEqual([
      'components/Fixture.tsx:1 (components) → @/pipeline [pipeline]',
      'Fixture.tsx:1 (root) → @/pipeline [pipeline]',
    ]);
  });

  it('keeps UI packages out of the framework-free layers', () => {
    expect(violationsIn([
      fixture('pipeline/fixture.ts', "import { useState } from 'react';"),
      fixture('renderer/fixture.ts', "import { create } from 'zustand';"),
      fixture('lib/fixture.ts', "import * as Dialog from '@radix-ui/react-dialog';"),
      fixture('gpu/fixture.ts', "import { createRoot } from 'react-dom/client';"),
      fixture('app/fixture.ts', "import { useState } from 'react';"),
      fixture('components/Fixture.tsx', "import { X } from 'lucide-react';"),
    ])).toEqual([
      'pipeline/fixture.ts:1 (pipeline) → react [package]',
      'renderer/fixture.ts:1 (renderer) → zustand [package]',
      'lib/fixture.ts:1 (lib) → @radix-ui/react-dialog [package]',
      'gpu/fixture.ts:1 (gpu) → react-dom/client [package]',
    ]);
  });

  it('ignores assets and modules outside src', () => {
    expect(violationsIn([
      fixture('lib/fixture.ts', [
        "import './fixture.css';",
        "import shader from '@/renderer/shaders/opendrt.wgsl?raw';",
        "import data from '../test/golden/baseline.json';",
      ].join('\n')),
      fixture('pipeline/decode/fixture.ts', "export const load = () => import('../../../wasm/rawloader/pkg/rawloader_wasm.js');"),
    ])).toEqual([]);
  });

  it('reports the line of the offending statement', () => {
    expect(violationsIn([
      fixture('lib/fixture.ts', [
        "import type { CfaType } from './types';",
        '',
        "import { useAppStore } from '@/app/store';",
      ].join('\n')),
    ])).toEqual(['lib/fixture.ts:3 (lib) → @/app/store [app]']);
  });
});
