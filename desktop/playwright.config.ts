import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'e2e', timeout: 600_000, workers: 1, retries: 0,
  use: { trace: 'retain-on-failure' },
  reporter: [['list'], ['json', { outputFile: process.env.XV_SMOKE_REPORT ?? 'test-results/smoke.json' }]],
});
