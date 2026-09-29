/** Conservative display probe shared by hosts without native luminance readings. */
export function mediaQueryHeadroom(hdr: boolean): {
  headroom: number;
  accurate: boolean;
} {
  return hdr
    ? { headroom: 2, accurate: false }
    : { headroom: 1, accurate: true };
}
