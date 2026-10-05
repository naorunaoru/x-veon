import { configDefaults, defineConfig } from 'vitest/config';
import path from 'node:path';
export default defineConfig({
  resolve: { alias: { '@': path.resolve(__dirname, '../shared/src') } },
  test: { exclude: [...configDefaults.exclude, 'src/**/*.native.test.ts'], globalSetup: ['./src/test/symlink-setup.ts'], environment: 'node', include: ['src/**/*.test.ts'] },
});
