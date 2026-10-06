import { defineConfig } from 'electron-vite';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import path from 'node:path';
import { cpSync, readdirSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { appIdentity } from './src/release/tags';
const shared = path.resolve(__dirname, '../shared');
const publicDir = path.join(shared, 'public');
const golden = process.env.XV_GOLDEN === '1';
const sha = execFileSync('git', ['rev-parse', '--short', 'HEAD']).toString().trim();
// An unset or empty XV_RELEASE_TAG is an untagged build. appIdentity itself rejects ''.
const identity = appIdentity(process.env.XV_RELEASE_TAG || undefined, sha, process.env.npm_lifecycle_event === 'dist');
const build = {
  channel: identity.channel,
  sha,
  date: new Date().toISOString().slice(0, 10),
  ...(identity.tag ? { tag: identity.tag, version: identity.version } : {}),
};
export default defineConfig({
  main: {
    plugins: [{
      name: 'xveon-release-identity', apply: 'build',
      // electron-builder packages exactly this identity: the bundle and its installer agree.
      writeBundle() { writeFileSync(path.resolve(__dirname, 'out/release.json'), JSON.stringify(identity, null, 2) + '\n'); },
    }],
    define: { __XV_BUILD__: JSON.stringify(build), __XV_GOLDEN__: JSON.stringify(golden) },
    resolve: { alias: { '@': path.join(shared, 'src') } },
    build: {
      rollupOptions: {
        input: {
          index: path.resolve(__dirname, 'src/main/index.ts'),
          worker: path.resolve(__dirname, 'src/worker/index.ts'),
        },
        // Native module paths resolve from __dirname; keep shared chunks beside worker.js.
        output: { chunkFileNames: '[name]-[hash].js' },
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
