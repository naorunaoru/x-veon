import { describe, it, expect, vi } from 'vitest';

// Stub out WASM-backed modules so the pure predicate can be imported in isolation.
vi.mock('@/app/hooks/useProcessing', () => ({ useProcessing: vi.fn() }));
vi.mock('@/app/store', () => ({ useAppStore: vi.fn() }));

import { shouldAutoProcess } from './useAutoProcess';

const queued = { status: 'queued' as const };
const done = { status: 'done' as const };

describe('shouldAutoProcess', () => {
  it('processes a queued file when initialized and idle', () => {
    expect(shouldAutoProcess(queued, true, false)).toBe(true);
  });
  it('does not process before init', () => {
    expect(shouldAutoProcess(queued, false, false)).toBe(false);
  });
  it('does not process while another job runs', () => {
    expect(shouldAutoProcess(queued, true, true)).toBe(false);
  });
  it('does not process a done file', () => {
    expect(shouldAutoProcess(done, true, false)).toBe(false);
  });
  it('does nothing without a file', () => {
    expect(shouldAutoProcess(undefined, true, false)).toBe(false);
  });
});
