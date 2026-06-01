// Extends Vitest's `expect` with jest-dom DOM matchers (toBeInTheDocument, etc.).
// Must use the /vitest subpath — the root import augments Jest, not Vitest.
import '@testing-library/jest-dom/vitest';
