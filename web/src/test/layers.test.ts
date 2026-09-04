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

const UI_PACKAGES = /^(react|react-dom|zustand|@radix-ui\/|lucide-react)(\/|$)/;
const FRAMEWORK_FREE: readonly Layer[] = ['lib', 'gpu', 'pipeline', 'renderer'];

function layerOf(absFile: string): Layer {
  const parts = relative(SRC, absFile).split(sep);
  const top = parts.length > 1 ? parts[0] : 'root';
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

function refsOf(file: string): Ref[] {
  const source = readFileSync(file, 'utf8');
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

describe('layer dependency rule', () => {
  const files = walk(SRC);

  it('sees the source tree', () => {
    expect(files.length).toBeGreaterThan(50);
  });

  it('has no violations', () => {
    const violations: string[] = [];
    for (const file of files) {
      const from = layerOf(file);
      for (const { spec, line } of refsOf(file)) {
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
    expect(violations).toEqual([]);
  });

  it('catches a side-effect import across layers', () => {
    const sourceFile = ts.createSourceFile(
      'x.ts',
      "import '@/app/store';",
      ts.ScriptTarget.Latest,
      true,
    );
    let spec = '';
    ts.forEachChild(sourceFile, (node) => {
      if (ts.isImportDeclaration(node) && ts.isStringLiteral(node.moduleSpecifier)) {
        spec = node.moduleSpecifier.text;
      }
    });
    expect(spec).toBe('@/app/store');
    expect(ALLOWED.lib.includes(layerOf(join(SRC, 'app/store.ts')))).toBe(false);
  });
});
