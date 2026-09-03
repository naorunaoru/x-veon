/// <reference types="vite/client" />

declare module '*.wgsl?raw' {
  const src: string;
  export default src;
}

interface Screen {
  highDynamicRangeHeadroom?: number;
}

/** Injected by vite.config.ts `define`; undefined under Vitest and in the Vite config itself. */
declare const __XV_BUILD__: { channel: 'stable' | 'beta' | 'dev'; sha: string; date: string } | undefined;
