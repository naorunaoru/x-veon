import { expect, it, vi } from 'vitest';
import { act } from '@testing-library/react';
import { getHost, setHost } from '@/app/services/host';
import { fakeHost } from '@/test/fake-host';
import { startApp } from './startApp';
const observed = vi.hoisted(() => ({ hosts: [] as unknown[] }));
vi.mock('./App', () => ({
  default: () => {
    observed.hosts.push(getHost());
    return null;
  },
}));
it('installs the supplied host before rendering App', async () => {
  setHost(null);
  const host = fakeHost();
  const root = document.createElement('div');
  let stop!: () => void;
  await act(async () => {
    stop = startApp(root, host);
  });
  expect(observed.hosts.length).toBeGreaterThan(0);
  expect(observed.hosts.every((value) => value === host)).toBe(true);
  await act(async () => stop());
});
