/// <reference types="vite/client" />

declare module '*.glsl?raw' {
  const src: string;
  export default src;
}

// Experimental Canvas HDR APIs (Chrome 122+)
interface WebGL2RenderingContext {
  drawingBufferStorage?(internalformat: number, width: number, height: number): void;
  drawingBufferColorSpace?: string;
  drawingBufferToneMapping?(options: { mode: string }): void;
}

interface HTMLCanvasElement {
  configureHighDynamicRange?(options: { mode: string }): void;
}

interface Screen {
  highDynamicRangeHeadroom?: number;
}
