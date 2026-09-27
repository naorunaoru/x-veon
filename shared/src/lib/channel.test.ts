import { describe, it, expect } from 'vitest';
import { basePath, storageNames, otherChannelLink, channelLabel, isChannel, BUILD, CHANNELS } from './channel';

describe('channel helpers', () => {
  it('maps channels to base paths', () => {
    expect(basePath('dev')).toBe('/');
    expect(basePath('stable')).toBe('/x-veon/');
    expect(basePath('beta')).toBe('/x-veon/beta/');
  });

  it('namespaces storage per channel', () => {
    expect(storageNames('stable')).toEqual({ dbName: 'xveon-stable', opfsRoot: 'stable' });
    expect(storageNames('beta')).toEqual({ dbName: 'xveon-beta', opfsRoot: 'beta' });
    expect(storageNames('dev')).toEqual({ dbName: 'xveon-dev', opfsRoot: 'dev' });
  });

  it('never reuses the frozen app storage names', () => {
    for (const c of CHANNELS) {
      expect(storageNames(c).dbName).not.toBe('xtrans-demosaic');
      expect(['raw', 'thumbnails', 'hwc-cache']).not.toContain(storageNames(c).opfsRoot);
    }
  });

  it('links stable and beta to each other, dev to nothing', () => {
    expect(otherChannelLink('stable')).toEqual({ channel: 'beta', label: 'beta', href: '/x-veon/beta/' });
    expect(otherChannelLink('beta')).toEqual({ channel: 'stable', label: 'stable', href: '/x-veon/' });
    expect(otherChannelLink('dev')).toBeNull();
  });

  it('labels channels for display', () => {
    expect(channelLabel('stable')).toBe('Stable');
    expect(channelLabel('beta')).toBe('Beta');
    expect(channelLabel('dev')).toBe('Dev');
  });

  it('validates channel strings', () => {
    expect(isChannel('beta')).toBe(true);
    expect(isChannel('prod')).toBe(false);
    expect(isChannel(undefined)).toBe(false);
  });

  it('falls back to a dev build stamp when no build define exists', () => {
    expect(BUILD).toEqual({ channel: 'dev', sha: 'test', date: '' });
  });
});
