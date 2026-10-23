import { expect, it } from 'vitest';
import { filterCounts, matchesFilter } from './rating';
import type { RatingFilter } from './rating';
import type { Rating } from '@/host';

it('matchesFilter table', () => {
  const table: [Rating, RatingFilter, boolean][] = [
    [0, 'all', true], [-1, 'star1', false], [1, 'star1', true], [2, 'star3', false],
    [5, 'star3', true], [-1, 'rejected', true], [0, 'rejected', false],
  ];
  expect(table.map(([rating, filter]) => [rating, filter, matchesFilter(rating, filter)])).toEqual(table);
});
it('filterCounts counts every rating against each filter', () => {
  expect(filterCounts([0, 1, 3, -1, 5])).toEqual({ all: 5, star1: 3, star2: 2, star3: 2, rejected: 1 });
});
