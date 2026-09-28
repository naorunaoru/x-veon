export interface HdrDisplayInfo {
  supported: boolean;
  headroom: number;
  accurate: boolean;
}
export interface DisplayReadings {
  currentEdr?: number;
  potentialEdr?: number;
  referenceEdr?: number;
  maxLuminance?: number;
  maxFullFrameLuminance?: number;
  minLuminance?: number;
  sdrWhite?: number;
  hdrEnabled?: boolean;
}
export interface DisplayHost {
  probe(): Promise<HdrDisplayInfo>;
  readings?(): Promise<DisplayReadings>;
  requestAccurateHeadroom?(): Promise<number | null>;
}
