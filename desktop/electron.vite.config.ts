import { defineConfig } from 'electron-vite';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import path from 'node:path';
import { cpSync, readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
const shared = path.resolve(__dirname, '../shared');
const publicDir = path.join(shared, 'public');
const golden = process.env.XV_GOLDEN === '1';
const build = {
  channel: process.env.npm_lifecycle_event === 'dist' ? 'beta' : 'dev',
  sha: execFileSync('git', ['rev-parse', '--short', 'HEAD']).toString().trim(),
  date: new Date().toISOString().slice(0, 10),
};
export default defineConfig({
  main: {
    define: { __XV_BUILD__: JSON.stringify(build), __XV_GOLDEN__: JSON.stringify(golden) },
    resolve: { alias: { '@': path.join(shared, 'src') } },
    build: {
      rollupOptions: {
        input: {
          index: path.resolve(__dirname, 'src/main/index.ts'),
          worker: path.resolve(__dirname, 'src/worker/index.ts'),
        },
      },
    },
  },
  preload: {
    resolve: { alias: { '@': path.join(shared, 'src') } },
    build: {
      rollupOptions: {
        input: path.resolve(__dirname, 'src/preload/index.ts'),
        output: { format: 'cjs', entryFileNames: 'index.js' },
      },
    },
  },
  renderer: {
    root: __dirname,
    base: '/',
    publicDir,
    plugins: [react(), wasm(), {
      name: 'copy-desktop-public-without-samples',
      apply: 'build',
      writeBundle(options) {
        if (golden || !options.dir) return;
        for (const entry of readdirSync(publicDir, { withFileTypes: true })) {
          if (entry.name === 'samples') continue;
          cpSync(path.join(publicDir, entry.name), path.join(options.dir, entry.name), { recursive: true });
        }
      },
    }],
    resolve: { alias: { '@': path.join(shared, 'src') } },
    define: { __XV_BUILD__: JSON.stringify(build), __XV_GOLDEN__: JSON.stringify(golden) },
    build: {
      target: 'esnext',
      copyPublicDir: golden,
      rollupOptions: { input: path.resolve(__dirname, 'index.html') },
    },
  },
});
