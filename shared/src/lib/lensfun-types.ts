/** LensFun data model (matches the JSON from convert-lensfun-db.ts) and the profile matched for a photo. */

export interface LfCamera {
  make: string;
  model: string;
  mount: string;
  cropfactor: number;
  variant?: string;
}

export interface LfDistortion {
  model: string;
  focal: number;
  k1?: number;
  k2?: number;
  a?: number;
  b?: number;
  c?: number;
}

export interface LfTCA {
  model: string;
  focal: number;
  vr?: number;
  vb?: number;
  br?: number;
  cr?: number;
  bb?: number;
  cb?: number;
  kr?: number;
  kb?: number;
}

export interface LfVignetting {
  model: string;
  focal: number;
  aperture: number;
  distance: number;
  k1: number;
  k2: number;
  k3: number;
}

export interface LfLens {
  make: string;
  model: string;
  aliases: string[];
  mounts: string[];
  cropfactor: number;
  /** Sensor aspect ratio of the calibration (width / height); absent in the data today, 3:2 then. */
  aspectRatio?: number;
  type?: string;
  distortion: LfDistortion[];
  tca: LfTCA[];
  vignetting: LfVignetting[];
}

/** A lens matched for a photo. `cropfactor` is the image's; the `calib*` fields describe the calibration sensor. */
export interface LensProfile {
  lensModel: string;
  mount: string;
  cropfactor: number;
  calibCropfactor: number;
  calibAspectRatio: number;
  distortion: LfDistortion[];
  tca: LfTCA[];
  vignetting: LfVignetting[];
}
