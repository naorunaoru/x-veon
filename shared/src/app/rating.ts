import type { Rating } from '@/host';

export type RatingFilter = 'all' | 'star1' | 'star2' | 'star3' | 'rejected';
export const RATING_FILTERS: readonly RatingFilter[] = ['all', 'star1', 'star2', 'star3', 'rejected'];
export const FILTER_LABELS: Record<RatingFilter, string> = { all: 'All', star1: '★1+', star2: '★2+', star3: '★3+', rejected: 'Rejected' };

/** `starN` keeps ratings of N or more (a reject, −1, never matches); `rejected` keeps only rejects. */
export function matchesFilter(rating: Rating, filter: RatingFilter): boolean {
  switch (filter) {
    case 'all': return true;
    case 'rejected': return rating === -1;
    case 'star1': return rating >= 1;
    case 'star2': return rating >= 2;
    case 'star3': return rating >= 3;
  }
}

/** How many of `ratings` each filter keeps. */
export function filterCounts(ratings: readonly Rating[]): Record<RatingFilter, number> {
  const counts = { all: 0, star1: 0, star2: 0, star3: 0, rejected: 0 };
  for (const rating of ratings) for (const filter of RATING_FILTERS) if (matchesFilter(rating, filter)) counts[filter]++;
  return counts;
}
