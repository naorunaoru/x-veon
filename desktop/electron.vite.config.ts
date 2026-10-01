import { defineConfig } from 'electron-vite';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
const shared = path.resolve(__dirname, '../shared');
const build = {
  channel: 'dev',
  sha: execFileSync('git', ['rev-parse', '--short', 'HEAD']).toString().trim(),
  date: new Date().toISOString().slice(0, 10),
};
export default defineConfig({
  main: {
    define: { __XV_BUILD__: JSON.stringify(build), __XV_GOLDEN__: JSON.stringify(process.env.XV_GOLDEN === '1') },
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
    publicDir: path.join(shared, 'public'),
    plugins: [react(), wasm()],
    resolve: { alias: { '@': path.join(shared, 'src') } },
    define: { __XV_BUILD__: JSON.stringify(build), __XV_GOLDEN__: JSON.stringify(process.env.XV_GOLDEN === '1') },
    build: {
      target: 'esnext',
      rollupOptions: { input: path.resolve(__dirname, 'index.html') },
    },
  },
});
