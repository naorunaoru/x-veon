import { defineConfig } from 'vitest/config';
import path from 'node:path';
export default defineConfig({
  resolve: { alias: { '@': path.resolve(__dirname, '../shared/src') } },
  test: { environment: 'node', include: ['src/**/*.native.test.ts'], testTimeout: 300_000, fileParallelism: false },
});
