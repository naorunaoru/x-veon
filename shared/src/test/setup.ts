// Extends Vitest's `expect` with jest-dom DOM matchers (toBeInTheDocument, etc.).
// Must use the /vitest subpath — the root import augments Jest, not Vitest.
import '@testing-library/jest-dom/vitest';
import { beforeEach, afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

// globals: false means @testing-library/react cannot detect `afterEach` at
// module evaluation time and therefore does not auto-register cleanup.
// We register it explicitly so the DOM is reset between tests.
afterEach(() => {
  cleanup();
});

// jsdom doesn't implement ResizeObserver; Radix UI's react-use-size requires it.
// A no-op stub is sufficient for component render/output tests that don't
// exercise actual resize callbacks.
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

import { setHost } from '@/app/services/host';
import { fakeHost } from './fake-host';
beforeEach(() => setHost(fakeHost()));
