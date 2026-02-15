/*
 * Standalone test harness for darktable's segmentation-based highlight
 * reconstruction.  Reads a binary fixture (from extract_fixture.py), runs the
 * algorithm, and writes the result as flat float32.
 *
 * Build:
 *   cc -O2 -o test_segbased test_segbased.c -lm
 *
 * Run:
 *   ./test_segbased fixture.bin output.bin
 */

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Stubs for darktable internals ---- */

typedef int gboolean;
#define TRUE 1
#define FALSE 0

typedef float dt_aligned_pixel_t[4];
#define DT_ALIGNED_ARRAY
#define DT_OMP_PRAGMA(...)
#define DT_OMP_FOR(...)
#define DT_OMP_DECLARE_SIMD(...)
#define for_each_channel(c, ...) for (int c = 0; c < 3; c++)
#define for_three_channels(c, ...) for (int c = 0; c < 3; c++)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static inline float sqrf(float a) { return a * a; }
static inline int feqf(float a, float b, float eps) {
  return fabsf(a - b) < eps;
}

#define dt_alloc_align_float(n) ((float *)calloc((n), sizeof(float)))
#define dt_alloc_align_int(n) ((int *)calloc((n), sizeof(int)))
#define dt_alloc_align_type(T, n) ((T *)calloc((n), sizeof(T)))
#define dt_calloc_aligned(sz) calloc(1, (sz))
#define dt_alloc_aligned(sz) malloc(sz)
#define dt_free_align(p) free(p)
#define dt_calloc_align_type(T, n) ((T *)calloc((n), sizeof(T)))
#define dt_print(...)

static inline size_t dt_round_size(size_t size, size_t align) {
  return (size + align - 1) & ~(align - 1);
}

/* CFA color lookup — supports both Bayer and X-Trans */
static uint8_t g_xtrans[6][6];
static uint32_t g_filters;

static inline int FC(size_t row, size_t col, uint32_t filters) {
  return filters >> (((row << 1 & 14) + (col & 1)) << 1) & 3;
}

static inline int FCNxtrans(int row, int col, const uint8_t (*xtrans)[6]) {
  return xtrans[(row + 600) % 6][(col + 600) % 6];
}

static inline int fcol(int row, int col, uint32_t filters,
                       const uint8_t (*xtrans)[6]) {
  if (filters == 9u)
    return FCNxtrans(row, col, xtrans);
  else
    return FC(row, col, filters);
}

static inline float scharr_gradient(const float *p, int w) {
  const float gx =
      47.0f / 255.0f * (p[-w - 1] - p[-w + 1] + p[w - 1] - p[w + 1]) +
      162.0f / 255.0f * (p[-1] - p[1]);
  const float gy =
      47.0f / 255.0f * (p[-w - 1] - p[w - 1] + p[-w + 1] - p[w + 1]) +
      162.0f / 255.0f * (p[-w] - p[w]);
  return sqrtf(sqrf(gx) + sqrf(gy));
}

/* ---- Include darktable's segmentation code directly ---- */
#include "segmentation.c"

/* ---- Constants from segbased.c ---- */
#define HL_RGB_PLANES 3
#define HL_BORDER 8
#define HL_POWERF 3.0f

/* ---- Core functions from segbased.c ---- */

static inline float _local_std_deviation(const float *p, const int w) {
  const int w2 = 2 * w;
  const float av =
      (p[-w2 - 1] + p[-w2] + p[-w2 + 1] + p[-w - 2] + p[-w - 1] + p[-w] +
       p[-w + 1] + p[-w + 2] + p[-2] + p[-1] + p[0] + p[1] + p[2] +
       p[w - 2] + p[w - 1] + p[w] + p[w + 1] + p[w + 2] + p[w2 - 1] +
       p[w2] + p[w2 + 1]) /
      21.0f;
  return sqrtf(
      (sqrf(p[-w2 - 1] - av) + sqrf(p[-w2] - av) + sqrf(p[-w2 + 1] - av) +
       sqrf(p[-w - 2] - av) + sqrf(p[-w - 1] - av) + sqrf(p[-w] - av) +
       sqrf(p[-w + 1] - av) + sqrf(p[-w + 2] - av) + sqrf(p[-2] - av) +
       sqrf(p[-1] - av) + sqrf(p[0] - av) + sqrf(p[1] - av) +
       sqrf(p[2] - av) + sqrf(p[w - 2] - av) + sqrf(p[w - 1] - av) +
       sqrf(p[w] - av) + sqrf(p[w + 1] - av) + sqrf(p[w + 2] - av) +
       sqrf(p[w2 - 1] - av) + sqrf(p[w2] - av) + sqrf(p[w2 + 1] - av)) /
          21.0f);
}

static float _calc_weight(const float *s, size_t loc, int w, float clipval) {
  const float smoothness =
      fmaxf(0.0f, 1.0f - 10.0f * sqrtf(_local_std_deviation(&s[loc], w)));
  float val = 0.0f;
  for (int y = -1; y < 2; y++)
    for (int x = -1; x < 2; x++)
      val += s[loc + y * w + x] / 9.0f;
  const float sval = fmaxf(1.0f, powf(fminf(clipval, val) / clipval, 2.0f));
  return sval * smoothness;
}

static void _calc_plane_candidates(const float *plane, const float *refavg,
                                   dt_iop_segmentation_t *seg, float clipval,
                                   float badlevel) {
  for (uint32_t id = 2; id < (uint32_t)seg->nr; id++) {
    seg->val1[id] = 0.0f;
    seg->val2[id] = 0.0f;
    if ((seg->ymax[id] - seg->ymin[id] > 2) &&
        (seg->xmax[id] - seg->xmin[id] > 2)) {
      size_t testref = 0;
      float testweight = 0.0f;
      for (int row = MAX(seg->border + 2, seg->ymin[id] - 2);
           row < MIN(seg->height - seg->border - 2, seg->ymax[id] + 3);
           row++) {
        for (int col = MAX(seg->border + 2, seg->xmin[id] - 2);
             col < MIN(seg->width - seg->border - 2, seg->xmax[id] + 3);
             col++) {
          const size_t pos = row * seg->width + col;
          const uint32_t sid = _get_segment_id(seg, pos);
          if ((sid == id) && (plane[pos] < clipval)) {
            const float wht =
                _calc_weight(plane, pos, seg->width, clipval) *
                ((seg->data[pos] & DT_SEG_ID_MASK) ? 1.0f : 0.75f);
            if (wht > testweight) {
              testweight = wht;
              testref = pos;
            }
          }
        }
      }
      if (testref && (testweight > 1.0f - badlevel)) {
        float sum = 0.0f, pix = 0.0f;
        const float weights[5][5] = {{1, 4, 6, 4, 1},
                                     {4, 16, 24, 16, 4},
                                     {6, 24, 36, 24, 6},
                                     {4, 16, 24, 16, 4},
                                     {1, 4, 6, 4, 1}};
        for (int y = -2; y < 3; y++) {
          for (int x = -2; x < 3; x++) {
            const size_t pos = testref + y * seg->width + x;
            if (plane[pos] < clipval) {
              sum += plane[pos] * weights[y + 2][x + 2];
              pix += weights[y + 2][x + 2];
            }
          }
        }
        const float av = sum / fmaxf(1.0f, pix);
        if (av > 0.125f * clipval) {
          seg->val1[id] = fminf(clipval, av);
          seg->val2[id] = refavg[testref];
        }
      }
    }
  }
}

static inline float _calc_refavg(const float *in, int row, int col, int width,
                                  int height) {
  dt_aligned_pixel_t mean = {0, 0, 0, 0};
  dt_aligned_pixel_t cnt = {0, 0, 0, 0};
  const int dymin = MAX(0, row - 1);
  const int dxmin = MAX(0, col - 1);
  const int dymax = MIN(height - 1, row + 1);
  const int dxmax = MIN(width - 1, col + 1);

  for (int dy = dymin; dy <= dymax; dy++) {
    for (int dx = dxmin; dx <= dxmax; dx++) {
      const float val = fmaxf(0.0f, in[dy * width + dx]);
      const int c =
          fcol(dy, dx, g_filters, (const uint8_t(*)[6])g_xtrans);
      mean[c] += val;
      cnt[c] += 1.0f;
    }
  }
  for_each_channel(c) mean[c] =
      (cnt[c] > 0.0f) ? powf(mean[c] / cnt[c], 1.0f / HL_POWERF) : 0.0f;

  const int color = fcol(row, col, g_filters, (const uint8_t(*)[6])g_xtrans);
  const dt_aligned_pixel_t croot_refavg = {0.5f * (mean[1] + mean[2]),
                                            0.5f * (mean[0] + mean[2]),
                                            0.5f * (mean[0] + mean[1]),
                                            0.0f};
  return croot_refavg[color];
}

static inline size_t _raw_to_plane(int pwidth, int row, int col) {
  return (HL_BORDER + (row / 3)) * pwidth + (col / 3) + HL_BORDER;
}

static void _masks_extend_border(float *mask, int width, int height,
                                 int border) {
  if (border <= 0)
    return;
  for (size_t row = border; row < (size_t)(height - border); row++) {
    size_t idx = row * width;
    for (size_t i = 0; i < (size_t)border; i++) {
      mask[idx + i] = mask[idx + border];
      mask[idx + width - i - 1] = mask[idx + width - border - 1];
    }
  }
  for (size_t col = 0; col < (size_t)width; col++) {
    float top = mask[border * width + MIN(width - border - 1, MAX((int)col, border))];
    float bot = mask[(height - border - 1) * width +
                     MIN(width - border - 1, MAX((int)col, border))];
    for (size_t i = 0; i < (size_t)border; i++) {
      mask[col + i * width] = top;
      mask[col + (height - i - 1) * width] = bot;
    }
  }
}

/* ---- Main ---- */

/* Fixture format:
 *   [0..3]   u32 width
 *   [4..7]   u32 height
 *   [8..11]  u32 period
 *   [12..15] u32 dy
 *   [16..19] u32 dx
 *   [20..]   u8[period*period] pattern
 *   [256..]  float32[width*height] CFA
 */

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <fixture.bin> <output.bin>\n", argv[0]);
    return 1;
  }

  FILE *fin = fopen(argv[1], "rb");
  if (!fin) {
    perror("open input");
    return 1;
  }

  /* Read header */
  uint8_t header[256];
  fread(header, 1, 256, fin);

  uint32_t width, height, period, dy_shift, dx_shift;
  memcpy(&width, header + 0, 4);
  memcpy(&height, header + 4, 4);
  memcpy(&period, header + 8, 4);
  memcpy(&dy_shift, header + 12, 4);
  memcpy(&dx_shift, header + 16, 4);

  uint8_t pattern_flat[36]; /* max 6x6 */
  memcpy(pattern_flat, header + 20, period * period);

  fprintf(stderr, "Fixture: %ux%u, period=%u, shift=dy%u dx%u\n", width,
          height, period, dy_shift, dx_shift);

  /* Set up CFA globals */
  if (period == 6) {
    /* X-Trans */
    g_filters = 9u;
    for (uint32_t y = 0; y < 6; y++)
      for (uint32_t x = 0; x < 6; x++)
        g_xtrans[y][x] = pattern_flat[y * 6 + x];
  } else {
    /* Bayer — encode into darktable's packed filter format */
    /* pattern_flat has the shifted pattern, we need to build filters uint32 */
    /* For simplicity, use X-Trans path with period 2 padded to 6 */
    /* Actually darktable's FC macro works differently.  Let's use xtrans path
       with the pattern tiled to 6x6 */
    g_filters = 9u; /* use xtrans path for generality */
    for (int y = 0; y < 6; y++)
      for (int x = 0; x < 6; x++)
        g_xtrans[y][x] = pattern_flat[(y % period) * period + (x % period)];
  }

  /* Read CFA data */
  size_t npix = (size_t)width * height;
  float *input = (float *)malloc(npix * sizeof(float));
  fread(input, sizeof(float), npix, fin);
  fclose(fin);

  /* Working copy (opposed reconstruction writes here, then segmentation
   * modifies it) */
  float *tmpout = (float *)malloc(npix * sizeof(float));
  memcpy(tmpout, input, npix * sizeof(float));

  const float clipval = 1.0f;
  const float cubeClip = powf(clipval, 1.0f / HL_POWERF);
  const float combineRadius = 2;
  const float badlevel = 0.5f;

  /* ---- Phase 1: Opposed inpainting (same as existing TS function) ---- */
  {
    const float chromLo = 0.2f;
    float chromSum[3] = {0}, chromCnt[3] = {0};

    for (int row = 1; row < (int)height - 1; row++) {
      for (int col = 1; col < (int)width - 1; col++) {
        float val = tmpout[row * width + col];
        if (val < chromLo || val >= clipval) continue;
        int ch = fcol((row + dy_shift), (col + dx_shift), g_filters,
                       (const uint8_t(*)[6])g_xtrans);

        dt_aligned_pixel_t mean = {0}, cnt_arr = {0};
        for (int ny = MAX(0, row - 1); ny <= MIN((int)height - 1, row + 1); ny++) {
          for (int nx = MAX(0, col - 1); nx <= MIN((int)width - 1, col + 1); nx++) {
            float v = fmaxf(0, tmpout[ny * width + nx]);
            int c = fcol((ny + dy_shift), (nx + dx_shift), g_filters,
                          (const uint8_t(*)[6])g_xtrans);
            mean[c] += v;
            cnt_arr[c] += 1;
          }
        }
        float cr[3];
        for (int c = 0; c < 3; c++)
          cr[c] = cnt_arr[c] > 0 ? cbrtf(mean[c] / cnt_arr[c]) : 0;
        float oppCr;
        if (ch == 0) oppCr = 0.5f * (cr[1] + cr[2]);
        else if (ch == 1) oppCr = 0.5f * (cr[0] + cr[2]);
        else oppCr = 0.5f * (cr[0] + cr[1]);
        float ref = oppCr * oppCr * oppCr;

        chromSum[ch] += val - ref;
        chromCnt[ch]++;
      }
    }

    float chrom[3] = {0};
    for (int c = 0; c < 3; c++)
      if (chromCnt[c] > 100) chrom[c] = chromSum[c] / chromCnt[c];

    for (int row = 1; row < (int)height - 1; row++) {
      for (int col = 1; col < (int)width - 1; col++) {
        int idx = row * width + col;
        if (tmpout[idx] < clipval) continue;
        int ch = fcol((row + dy_shift), (col + dx_shift), g_filters,
                       (const uint8_t(*)[6])g_xtrans);

        dt_aligned_pixel_t mean = {0}, cnt_arr = {0};
        for (int ny = MAX(0, row - 1); ny <= MIN((int)height - 1, row + 1); ny++) {
          for (int nx = MAX(0, col - 1); nx <= MIN((int)width - 1, col + 1); nx++) {
            float v = fmaxf(0, tmpout[ny * width + nx]);
            int c = fcol((ny + dy_shift), (nx + dx_shift), g_filters,
                          (const uint8_t(*)[6])g_xtrans);
            mean[c] += v;
            cnt_arr[c] += 1;
          }
        }
        float cr[3];
        for (int c = 0; c < 3; c++)
          cr[c] = cnt_arr[c] > 0 ? cbrtf(mean[c] / cnt_arr[c]) : 0;
        float oppCr;
        if (ch == 0) oppCr = 0.5f * (cr[1] + cr[2]);
        else if (ch == 1) oppCr = 0.5f * (cr[0] + cr[2]);
        else oppCr = 0.5f * (cr[0] + cr[1]);
        float ref = oppCr * oppCr * oppCr;

        tmpout[idx] = fmaxf(tmpout[idx], ref + chrom[ch]);
      }
    }

    fprintf(stderr, "Opposed pass done. chrom=[%.4f, %.4f, %.4f]\n",
            chrom[0], chrom[1], chrom[2]);
  }

  /* ---- Phase 2: Segmentation-based reconstruction ---- */
  {
    /* Note: darktable uses (row + dy_shift) in fcol for the shifted pattern.
       Our fixture has the pattern already encoded with the shift in the xtrans
       table, so fcol(row+dy, col+dx, ...) gives the correct channel. */

    const size_t pwidth = dt_round_size(width / 3, 2) + 2 * HL_BORDER;
    const size_t pheight = dt_round_size(height / 3, 2) + 2 * HL_BORDER;
    const size_t p_size = dt_round_size(pwidth * pheight, 64);

    float *plane[HL_RGB_PLANES];
    float *refavg[HL_RGB_PLANES];
    for (int i = 0; i < HL_RGB_PLANES; i++) {
      plane[i] = dt_alloc_align_float(p_size);
      refavg[i] = dt_alloc_align_float(p_size);
    }

    const int maxsegs = MAX(256, (int)(width * height) / 4000);
    dt_iop_segmentation_t segs[HL_RGB_PLANES];
    for (int i = 0; i < HL_RGB_PLANES; i++)
      dt_segmentation_init_struct(&segs[i], pwidth, pheight, HL_BORDER + 1,
                                  maxsegs);

    /* xshifter: for Bayer with G at (0,0), use 1; else 2.
       This centres the 3×3 box on green for better chroma stability. */
    const int ch00 = fcol(dy_shift, dx_shift, g_filters,
                           (const uint8_t(*)[6])g_xtrans);
    const int xshifter = (period == 2 && ch00 == 1) ? 1 : 2;

    /* Populate planes from 3x3 superpixels */
    int anyclipped = 0;
    for (int row = 1; row < (int)height - 1; row++) {
      for (int col = 1; col < (int)width - 1; col++) {
        if (col % 3 != xshifter || row % 3 != 1) continue;

        dt_aligned_pixel_t mean = {0}, cnt_arr = {0};
        for (int dy = row - 1; dy < row + 2; dy++) {
          for (int dx = col - 1; dx < col + 2; dx++) {
            float val = tmpout[dy * width + dx];
            int c = fcol((dy + dy_shift), (dx + dx_shift), g_filters,
                          (const uint8_t(*)[6])g_xtrans);
            mean[c] += val;
            cnt_arr[c] += 1;
          }
        }

        for_each_channel(c) mean[c] =
            (cnt_arr[c] > 0) ? powf(mean[c] / cnt_arr[c], 1.0f / HL_POWERF)
                             : 0.0f;

        const dt_aligned_pixel_t cube_refavg = {0.5f * (mean[1] + mean[2]),
                                                 0.5f * (mean[0] + mean[2]),
                                                 0.5f * (mean[0] + mean[1]),
                                                 0.0f};

        const size_t o = _raw_to_plane(pwidth, row, col);
        for_three_channels(c) {
          plane[c][o] = mean[c];
          refavg[c][o] = cube_refavg[c];
          if (mean[c] > cubeClip) {
            segs[c].data[o] = 1;
            anyclipped++;
          }
        }
      }
    }

    fprintf(stderr, "Clipped superpixels: %d\n", anyclipped);

    if (anyclipped >= 20) {
      for (int i = 0; i < HL_RGB_PLANES; i++)
        _masks_extend_border(plane[i], pwidth, pheight, HL_BORDER);

      for (int p = 0; p < HL_RGB_PLANES; p++)
        dt_segments_combine(&segs[p], (int)combineRadius);

      for (int p = 0; p < HL_RGB_PLANES; p++)
        dt_segmentize_plane(&segs[p]);

      for (int p = 0; p < HL_RGB_PLANES; p++)
        _calc_plane_candidates(plane[p], refavg[p], &segs[p], cubeClip,
                               badlevel);

      fprintf(stderr, "Segments: R=%d G=%d B=%d\n", segs[0].nr - 2,
              segs[1].nr - 2, segs[2].nr - 2);

      /* Dump candidates */
      for (int p = 0; p < HL_RGB_PLANES; p++) {
        for (int id = 2; id < segs[p].nr; id++) {
          if (segs[p].val1[id] != 0.0f)
            fprintf(stderr, "  [%c] seg %d: val1=%.6f val2=%.6f\n",
                    "RGB"[p], id, segs[p].val1[id], segs[p].val2[id]);
        }
      }

      /* Reconstruct */
      for (int row = 1; row < (int)height - 1; row++) {
        for (int col = 1; col < (int)width - 1; col++) {
          const size_t idx = row * width + col;
          const float inval = fmaxf(0.0f, input[idx]);
          if (inval < clipval) continue;

          const int color = fcol((row + dy_shift), (col + dx_shift),
                                  g_filters, (const uint8_t(*)[6])g_xtrans);
          const size_t o = _raw_to_plane(pwidth, row, col);
          const uint32_t pid = _get_segment_id(&segs[color], o);

          if (pid > 1 && pid < (uint32_t)segs[color].nr) {
            const float candidate = segs[color].val1[pid];
            if (candidate != 0.0f) {
              const float cand_reference = segs[color].val2[pid];
              /* Compute refavg at this location in cube-root space */
              dt_aligned_pixel_t lmean = {0}, lcnt = {0};
              for (int ny = MAX(0, row - 1); ny <= MIN((int)height - 1, row + 1); ny++) {
                for (int nx = MAX(0, col - 1); nx <= MIN((int)width - 1, col + 1); nx++) {
                  float v = fmaxf(0, input[ny * width + nx]);
                  int c = fcol((ny + dy_shift), (nx + dx_shift), g_filters,
                                (const uint8_t(*)[6])g_xtrans);
                  lmean[c] += v;
                  lcnt[c] += 1;
                }
              }
              for_each_channel(c) lmean[c] =
                  (lcnt[c] > 0) ? powf(lmean[c] / lcnt[c], 1.0f / HL_POWERF)
                                : 0.0f;
              const float cr_refavg[3] = {
                  0.5f * (lmean[1] + lmean[2]),
                  0.5f * (lmean[0] + lmean[2]),
                  0.5f * (lmean[0] + lmean[1])};
              const float refavg_here = cr_refavg[color];

              const float oval =
                  powf(refavg_here + candidate - cand_reference, HL_POWERF);
              tmpout[idx] = fmaxf(inval, oval);
            }
          }
        }
      }
    }

    for (int i = 0; i < HL_RGB_PLANES; i++) {
      dt_segmentation_free_struct(&segs[i]);
      free(plane[i]);
      free(refavg[i]);
    }
  }

  /* Write output */
  FILE *fout = fopen(argv[2], "wb");
  fwrite(tmpout, sizeof(float), npix, fout);
  fclose(fout);

  fprintf(stderr, "Output written to %s\n", argv[2]);

  free(input);
  free(tmpout);
  return 0;
}
