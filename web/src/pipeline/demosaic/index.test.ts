import { describe, it, expect } from 'vitest';
import { DEMOSAIC_STRATEGIES, strategyFor } from './index';
import { DEMOSAIC_METHODS } from '@/lib/catalog';

describe('demosaic strategy registry', () => {
  it('has exactly one strategy per catalogue method', () => {
    const strategyIds = DEMOSAIC_STRATEGIES.map((s) => s.id).sort();
    const catalogueIds = DEMOSAIC_METHODS.map((m) => m.id).sort();
    expect(strategyIds).toEqual(catalogueIds);
    expect(new Set(strategyIds).size).toBe(strategyIds.length);
  });

  it('looks a strategy up by id and rejects unknown ids', () => {
    expect(strategyFor('dht').id).toBe('dht');
    expect(strategyFor('neural-net').id).toBe('neural-net');
    expect(() => strategyFor('nope' as never)).toThrow(/unknown demosaic method/);
  });
});
