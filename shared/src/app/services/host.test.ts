import { afterEach, expect, it } from 'vitest';
import { getHost, setHost } from './host';
import { fakeHost } from '@/test/fake-host';
afterEach(() => setHost(fakeHost()));
it('requires explicit host installation and returns that instance', () => {
  setHost(null);
  expect(() => getHost()).toThrow('host not installed');
  const host = fakeHost();
  setHost(host);
  expect(getHost()).toBe(host);
  expect(host.library.remove).toBeUndefined();
  expect(host.library.clear).toBeUndefined();
  expect(host.display.requestAccurateHeadroom).toBeUndefined();
});
