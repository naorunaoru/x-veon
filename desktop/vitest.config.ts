import { defineConfig } from 'vitest/config';
import path from 'node:path';
export default defineConfig({
  resolve: { alias: { '@': path.resolve(__dirname, '../shared/src') } },
  test: { globalSetup: ['./src/test/symlink-setup.ts'], environment: 'node', include: ['src/**/*.test.ts'] },
});
