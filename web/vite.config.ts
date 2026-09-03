import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import tailwindcss from '@tailwindcss/vite';
import path from 'path';
import { execSync } from 'node:child_process';
import { basePath, isChannel, CHANNELS, type Channel } from './src/lib/channel';

/**
 * XV_CHANNEL decides the base path, the storage namespace and the build stamp.
 *   stable → /x-veon/        beta → /x-veon/beta/        dev → /
 * It is required for `vite build` and `vite preview` (an accidental default could
 * produce a wrongly-based site); the dev server defaults to `dev`.
 */
function resolveChannel(command: 'build' | 'serve', isPreview: boolean): Channel {
  const raw = process.env.XV_CHANNEL;
  if (raw === undefined) {
    if (command === 'build' || isPreview) {
      throw new Error(`XV_CHANNEL is required for vite build / vite preview (one of: ${CHANNELS.join(', ')})`);
    }
    return 'dev';
  }
  if (!isChannel(raw)) {
    throw new Error(`XV_CHANNEL must be one of ${CHANNELS.join(', ')}; got "${raw}"`);
  }
  return raw;
}

function buildSha(): string {
  if (process.env.XV_BUILD_SHA) return process.env.XV_BUILD_SHA.slice(0, 7);
  try {
    return execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
  } catch {
    return 'local';
  }
}

export default defineConfig(({ command, isPreview }) => {
  const channel = resolveChannel(command, isPreview ?? false);
  const build = { channel, sha: buildSha(), date: new Date().toISOString().slice(0, 10) };
  return {
    base: basePath(channel),
    define: {
      __XV_BUILD__: JSON.stringify(build),
    },
    plugins: [react(), wasm(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    optimizeDeps: {
      exclude: ['onnxruntime-web'],
    },
    server: {
      headers: {
        // Required for SharedArrayBuffer (ONNX Runtime WASM fallback)
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cross-Origin-Opener-Policy': 'same-origin',
      },
    },
    preview: {
      // GitHub Pages sends no COOP/COEP; mirror that so `vite preview` behaves like the deployed site
      // (with the dev-server headers the page is cross-origin isolated and ONNX Runtime's worker
      // threads try to load the app bundle and never initialise).
      headers: {},
    },
    build: {
      target: 'esnext',
    },
  };
});
