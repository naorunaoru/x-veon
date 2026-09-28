import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import path from 'node:path';
export default defineConfig({
  plugins: [react(), wasm()],
  resolve: { alias: { '@': path.resolve(__dirname, '../shared/src') } },
  test: { environment: 'jsdom', setupFiles: ['./src/test/setup.ts'], include: ['src/**/*.test.{ts,tsx}'] },
});
