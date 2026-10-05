#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Evaluate exported demosaic models against truth that no demosaicer touched.

Truth is real RGB from X-Trans RAFs: each colour averaged over 6x6 CFA cells. The mosaic
the model sees is simulated from it, for X-Trans or for Bayer.

    python tools/eval_truth.py MODEL.onnx [MODEL.onnx ...] --raf-dir DIR --cfa xtrans|bayer [--report out.json]

Models with the five-channel input are the subject. Models with v6's two inputs
(mosaic + white balance) are accepted for comparison and get the camera's white balance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cfa import CFA_REGISTRY, find_pattern_shift, make_channel_masks, make_model_input  # noqa: E402

TILE = 288
OVERLAP = 24                 # the app's tile overlap
INTERIOR = 32                # S reaches up to 27 px; closer to a tile edge than this, padding shows
FLOOR = 1e-4                 # model.MEAN_FLOOR
TILES_PER_IMAGE = 4
GREY = np.array([0.6, 1.0, 0.7], dtype=np.float32)   # raw-space colour of a neutral patch
# Seam measurement: tile origins in a 552 px strip. All multiples of 12, so every tile sees
# the same CFA layout and the same packing and halving phase as in the app.
SEAM_A, SEAM_B, SEAM_NATURAL_REF, SEAM_SHADOW_REF = 0, 264, 132, 216
SEAM_BAND = (256, 296)
SHADOW_FROM, SHADOW_FACTOR = 216, 32.0

Predictor = Callable[[np.ndarray, np.ndarray], np.ndarray]   # (N,3,T,T) truth, (N,3) camera wb -> (N,3,T,T)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def blend_weights(patch: int = TILE, overlap: int = OVERLAP) -> np.ndarray:
    """The app's 1-D tile blend weights (blendWeights1d in tile-blend-gpu.ts)."""
    w = np.ones(patch, dtype=np.float32)
    for i in range(overlap):
        w[i] = w[patch - 1 - i] = (i + 1) / (overlap + 1)
    return w


def accuracy_db(pred: np.ndarray, truth: np.ndarray) -> float:
    """-20 log10(RMSE) of two arrays of display-encoded values."""
    mse = float(np.mean((pred.astype(np.float64) - truth.astype(np.float64)) ** 2))
    return -10.0 * math.log10(max(mse, 1e-20))


def ripple_percent(values: np.ndarray, levels: np.ndarray) -> float:
    """Variation of a (3, H, W) output over a uniform patch, as % of each channel's level.

    The plain standard deviation, largest channel. No period is assumed: averaging by 6x6
    phase would hide a four-pixel pattern, which Bayer S can produce.
    """
    return float(max(100.0 * values[c].std() / levels[c] for c in range(3)))


def eligible(mosaic_means: np.ndarray) -> np.ndarray:
    """Tiles whose mosaic mean, the quantity the model itself computes, is at or above the floor."""
    return mosaic_means >= FLOOR


def bilinear(truth: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Reference demosaic: 5x5 tent-weighted normalised convolution, measured samples kept."""
    k1 = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    n, _, h, w = truth.shape
    out = np.empty_like(truth)
    for c in range(3):
        v = np.pad(truth[:, c] * masks[c], ((0, 0), (2, 2), (2, 2)))
        m = np.pad(masks[c], 2)
        num = np.zeros((n, h, w), np.float32)
        den = np.zeros((h, w), np.float32)
        for i in range(5):
            for j in range(5):
                num += k1[i] * k1[j] * v[:, i:i + h, j:j + w]
                den += k1[i] * k1[j] * m[i:i + h, j:j + w]
        out[:, c] = np.where(masks[c] > 0, truth[:, c], num / den)
    return out


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------

@dataclass
class Source:
    path: str
    rgb: np.ndarray          # (H, W, 3) true RGB at 1/6 scale, raw units
    display_wb: np.ndarray   # (3,) frozen: grey-world gains from the unmodified truth image
    display_gain: float      # frozen: 0.18 / mean green
    camera_wb: np.ndarray    # (3,) camera white balance, green = 1 (for two-input models)
    black: float
    white: float


@dataclass
class Tile:
    source: int
    y: int
    x: int


def load_source(path: str) -> Source:
    import rawpy

    with rawpy.imread(path) as raw:
        mosaic = raw.raw_image_visible.astype(np.float32)
        dy, dx = find_pattern_shift(raw.raw_colors_visible[:6, :6].copy(), CFA_REGISTRY["xtrans"])
        black, white = float(raw.black_level_per_channel[0]), float(raw.white_level)
        wb = np.array(raw.camera_whitebalance[:3], dtype=np.float32)
    mosaic = mosaic[dy:, dx:]                                   # canonical pattern at the origin
    h, w = mosaic.shape[0] // 6 * 6, mosaic.shape[1] // 6 * 6
    mosaic = (mosaic[:h, :w] - black) / (white - black)
    cells = mosaic.reshape(h // 6, 6, w // 6, 6).transpose(0, 2, 1, 3).reshape(h // 6, w // 6, 36)
    flat = CFA_REGISTRY["xtrans"].reshape(36)
    rgb = np.stack([cells[:, :, flat == k].mean(axis=2) for k in range(3)], axis=2)
    g = float(rgb[..., 1].mean())
    display_wb = np.array([g / rgb[..., 0].mean(), 1.0, g / rgb[..., 2].mean()], dtype=np.float32)
    return Source(path, rgb.astype(np.float32), display_wb, 0.18 / g, wb / wb[1], black, white)


def load_sources(paths: list[str], loader: Callable[[str], Source] = load_source) -> list[Source]:
    """Load every RAF that is X-Trans; skip the others with a note."""
    sources = []
    for path in paths:
        try:
            sources.append(loader(path))
        except ValueError as e:
            print(f"skipped {path}: {e}")
    return sources


def pick_tiles(sources: list[Source]) -> list[Tile]:
    """Up to four 288 px tiles per image, the most detailed by horizontal green difference."""
    tiles: list[Tile] = []
    for si, src in enumerate(sources):
        h, w = src.rgb.shape[:2]
        cand = [(y, x) for y in range(0, h - TILE + 1, TILE) for x in range(0, w - TILE + 1, TILE)]
        cand.sort(key=lambda yx: -float(np.abs(np.diff(src.rgb[yx[0]:yx[0] + TILE, yx[1]:yx[1] + TILE, 1], axis=1)).mean()))
        tiles += [Tile(si, y, x) for y, x in cand[:TILES_PER_IMAGE]]
    return tiles


class Evaluation:
    """Truth tiles plus the frozen display transform of the image each came from."""

    def __init__(self, sources: list[Source], cfa_type: str) -> None:
        self.sources = sources
        self.cfa_type = cfa_type
        self.tiles = pick_tiles(sources)
        self.masks = make_channel_masks(TILE, TILE, CFA_REGISTRY[cfa_type]).numpy()
        self.truth = np.stack([self.cut(t.source, t.y, t.x, TILE) for t in self.tiles])
        self.wb = np.stack([sources[t.source].display_wb for t in self.tiles]).reshape(-1, 3, 1, 1)
        self.gain = np.array([sources[t.source].display_gain for t in self.tiles], np.float32).reshape(-1, 1, 1, 1)
        self.camera_wb = np.stack([sources[t.source].camera_wb for t in self.tiles])

    def cut(self, source: int, y: int, x: int, width: int) -> np.ndarray:
        tile: np.ndarray = np.clip(self.sources[source].rgb[y:y + TILE, x:x + width].transpose(2, 0, 1), None, 1.0)
        return tile

    def display(self, values: np.ndarray, rows: slice | np.ndarray = slice(None)) -> np.ndarray:
        """White balance, exposure and gamma, with the parameters frozen per source image."""
        shown: np.ndarray = np.clip(values * self.wb[rows] * self.gain[rows], 0.0, 1.0) ** (1 / 2.2)
        return shown

    def mosaic_means(self, truth: np.ndarray) -> np.ndarray:
        means: np.ndarray = (np.clip(truth, None, 1.0) * self.masks).sum(axis=1).mean(axis=(1, 2))
        return means


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_predictor(path: str, masks: np.ndarray) -> tuple[Predictor, str]:
    """Wrap an ONNX model. Returns (predictor, contract) with contract 'five-channel' or 'mosaic+wb'."""
    import onnxruntime as ort

    session = ort.InferenceSession(path)
    inputs = session.get_inputs()
    masks_t = torch.from_numpy(masks)
    if len(inputs) == 1 and inputs[0].shape[1] == 5:
        contract = "five-channel"
    elif len(inputs) == 2 and inputs[0].shape[1] == 1:
        contract = "mosaic+wb"
    else:
        raise SystemExit(f"{path}: unsupported inputs {[(i.name, i.shape) for i in inputs]}")

    def predict(truth: np.ndarray, camera_wb: np.ndarray) -> np.ndarray:
        mosaic = (np.clip(truth, None, 1.0) * masks).sum(axis=1, keepdims=True).astype(np.float32)
        out = []
        for i in range(0, len(mosaic), 8):
            if contract == "five-channel":
                feed = {inputs[0].name: make_model_input(torch.from_numpy(mosaic[i:i + 8]), masks_t).numpy()}
            else:
                feed = {inputs[0].name: mosaic[i:i + 8], inputs[1].name: camera_wb[i:i + 8].astype(np.float32)}
            out.append(session.run(None, feed)[0])
        return np.concatenate(out)

    return predict, contract


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

IN = slice(INTERIOR, TILE - INTERIOR)


def measure_as_shot(ev: Evaluation, predict: Predictor) -> dict:
    out = predict(ev.truth, ev.camera_wb)
    ref = bilinear(ev.truth, ev.masks)
    t = ev.display(ev.truth)[..., IN, IN]
    return {"model_db": accuracy_db(ev.display(out)[..., IN, IN], t),
            "bilinear_db": accuracy_db(ev.display(ref)[..., IN, IN], t), "prediction": out}


def measure_exposure(ev: Evaluation, predict: Predictor, as_shot: np.ndarray) -> list[dict]:
    """Each exposure is compared with the as-shot accuracy of exactly the same tiles."""
    rows = []
    for stops in (-2, -4, -6):
        k = 2.0 ** stops
        ok = eligible(ev.mosaic_means(ev.truth * k))
        out = predict(ev.truth * k, ev.camera_wb) / k
        row: dict = {"ev": stops, "eligible_tiles": int(ok.sum()), "below_floor_tiles": int((~ok).sum())}
        for name, sel in (("eligible", ok), ("below_floor", ~ok)):
            if sel.any():
                t = ev.display(ev.truth[sel], sel)[..., IN, IN]
                row[f"{name}_db"] = accuracy_db(ev.display(out[sel], sel)[..., IN, IN], t)
                row[f"{name}_as_shot_db"] = accuracy_db(ev.display(as_shot[sel], sel)[..., IN, IN], t)
        rows.append(row)
    return rows


def measure_shadows(ev: Evaluation, as_shot: np.ndarray) -> dict:
    """Model error over bilinear error, by brightness relative to the tile's own mean."""
    truth = ev.truth[..., IN, IN]
    err_m = (as_shot - ev.truth)[..., IN, IN]
    err_b = (bilinear(ev.truth, ev.masks) - ev.truth)[..., IN, IN]
    stops = np.log2(np.maximum(truth.mean(axis=1, keepdims=True), 1e-12)
                    / ev.truth.mean(axis=(1, 2, 3), keepdims=True))
    result = {}
    for name, sel in (("more_than_3_stops_below", stops < -3), ("within_1_stop", np.abs(stops) <= 1)):
        sel3 = np.broadcast_to(sel, err_m.shape)
        n = int(sel3.sum())
        rmse_b = float(np.sqrt(np.mean(err_b[sel3] ** 2))) if n else 0.0
        measurable = n >= 50_000 and rmse_b >= 1e-6
        result[name] = {"values": n, "measurable": measurable,
                        "ratio": float(np.sqrt(np.mean(err_m[sel3] ** 2))) / rmse_b if measurable else None}
    return result


def measure_half_tile(ev: Evaluation, predict: Predictor, as_shot: np.ndarray) -> dict:
    """Accuracy lost in the untouched half when the other half is made 3 stops brighter or darker."""
    result = {}
    scored = {"left": slice(INTERIOR, 144 - INTERIOR), "right": slice(144 + INTERIOR, TILE - INTERIOR)}
    for label, factor in (("brighter", 8.0), ("darker", 1 / 8)):
        ref_px: list[np.ndarray] = []
        test_px: list[np.ndarray] = []
        truth_px: list[np.ndarray] = []
        for modified, keep in (("left", "right"), ("right", "left")):
            changed = ev.truth.copy()
            cols = slice(0, 144) if modified == "left" else slice(144, TILE)
            changed[..., cols] = np.clip(changed[..., cols] * factor, None, 1.0)
            out = predict(changed, ev.camera_wb)
            for store, values in ((ref_px, as_shot), (test_px, out), (truth_px, ev.truth)):
                store.append(ev.display(values)[..., IN, scored[keep]])
        t = np.concatenate(truth_px)
        result[label] = accuracy_db(np.concatenate(ref_px), t) - accuracy_db(np.concatenate(test_px), t)
    return result


def measure_seam(ev: Evaluation, predict: Predictor) -> dict:
    """Accuracy lost by the app's blended tiles in the band around a seam."""
    strips = [t for t in ev.tiles if t.x + 552 <= ev.sources[t.source].rgb.shape[1]]
    if not strips:
        return {"strips": 0}
    rows = np.array([i for i, t in enumerate(ev.tiles) if t in strips])
    strip = np.stack([ev.cut(t.source, t.y, t.x, 552) for t in strips])
    cam = ev.camera_wb[rows]
    w = blend_weights()
    b0, b1 = SEAM_BAND

    def tile(values: np.ndarray, origin: int) -> np.ndarray:
        return predict(np.ascontiguousarray(values[..., origin:origin + TILE]), cam)

    def blended(values: np.ndarray) -> np.ndarray:
        acc = np.zeros(values.shape, np.float32)
        wsum = np.zeros(552, np.float32)
        for origin in (SEAM_A, SEAM_B):
            acc[..., origin:origin + TILE] += tile(values, origin) * w
            wsum[origin:origin + TILE] += w
        return (acc / wsum)[..., IN, b0:b1]

    def band(values: np.ndarray) -> np.ndarray:
        return ev.display(values, rows)[..., :, :]

    truth_band = ev.display(strip[..., IN, b0:b1], rows)
    natural_ref = tile(strip, SEAM_NATURAL_REF)[..., IN, b0 - SEAM_NATURAL_REF:b1 - SEAM_NATURAL_REF]
    natural = accuracy_db(band(natural_ref), truth_band) - accuracy_db(band(blended(strip)), truth_band)

    shadow = strip.copy()
    shadow[..., SHADOW_FROM:] /= SHADOW_FACTOR                       # deep shadow from column 216 on
    shadow_ref = tile(shadow, SEAM_SHADOW_REF)[..., IN, b0 - SEAM_SHADOW_REF:b1 - SEAM_SHADOW_REF]
    lost = (accuracy_db(band(shadow_ref * SHADOW_FACTOR), truth_band)
            - accuracy_db(band(blended(shadow) * SHADOW_FACTOR), truth_band))
    return {"strips": len(strips), "natural_lost_db": natural, "bright_beside_shadow_lost_db": lost}


def measure_ripple(ev: Evaluation, predict: Predictor) -> dict:
    """Output variation for a uniform grey patch, % of its level."""
    grey_wb = (1.0 / GREY) / (1.0 / GREY)[1]
    cam = grey_wb.reshape(1, 3).astype(np.float32)

    def run(level: float, bright_left: float | None, region: tuple[slice, slice]) -> float:
        patch = np.broadcast_to((GREY * level).reshape(1, 3, 1, 1), (1, 3, TILE, TILE)).copy()
        if bright_left is not None:
            patch[..., :144] = (GREY * bright_left).reshape(1, 3, 1, 1)
        return ripple_percent(predict(patch, cam)[0][:, region[0], region[1]], GREY * level)

    whole = (slice(48, TILE - 48), slice(48, TILE - 48))
    return {"patch_at_0.1": run(0.1, None, whole), "patch_at_0.003": run(0.003, None, whole),
            "dark_in_bright_tile": run(0.003, 0.2, (slice(48, TILE - 48), slice(144 + 48, TILE - 48)))}


def evaluate(path: str, ev: Evaluation) -> dict:
    predict, contract = load_predictor(path, ev.masks)
    shot = measure_as_shot(ev, predict)
    prediction = shot.pop("prediction")
    return {"model": path, "contract": contract, "as_shot": shot,
            "exposure": measure_exposure(ev, predict, prediction),
            "shadows_inside_a_tile": measure_shadows(ev, prediction),
            "half_tile_lost_db": measure_half_tile(ev, predict, prediction),
            "seam": measure_seam(ev, predict),
            "ripple_percent": measure_ripple(ev, predict)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="ONNX files")
    parser.add_argument("--raf-dir", action="append", required=True,
                        help="Folder of X-Trans RAFs, searched recursively; repeatable")
    parser.add_argument("--cfa", choices=["xtrans", "bayer"], required=True, help="Mosaic to simulate")
    parser.add_argument("--report", default=None, help="Write sources, tiles, settings and scores as JSON")
    args = parser.parse_args()

    paths = sorted(str(p) for d in args.raf_dir for p in Path(d).rglob("*") if p.suffix.lower() == ".raf")
    if not paths:
        raise SystemExit("no RAF files found")
    sources = load_sources(paths)
    if not sources:
        raise SystemExit("no X-Trans RAF among the files found")
    ev = Evaluation(sources, args.cfa)
    print(f"{len(ev.tiles)} tiles from {len(sources)} RAFs, mosaic: {args.cfa}")
    results = [evaluate(m, ev) for m in args.models]
    for r in results:
        s = r["as_shot"]
        print(f"\n{r['model']} ({r['contract']})")
        print(f"  as shot: {s['model_db']:.2f} dB (bilinear {s['bilinear_db']:.2f} dB)")
        for e in r["exposure"]:
            line = f"  {e['ev']:+d} EV: "
            if "eligible_db" in e:
                line += f"{e['eligible_db']:.2f} dB against {e['eligible_as_shot_db']:.2f} dB as shot on the same {e['eligible_tiles']} tiles"
            if e["below_floor_tiles"]:
                line += f"; {e['below_floor_tiles']} tiles below the floor: {e.get('below_floor_db', float('nan')):.2f} dB"
            print(line)
        for name, b in r["shadows_inside_a_tile"].items():
            print(f"  shadows, {name}: " + (f"R = {b['ratio']:.3f}" if b["measurable"] else "not measurable") + f" ({b['values']} values)")
        print("  half tile lost: " + ", ".join(f"{k} {v:+.3f} dB" for k, v in r["half_tile_lost_db"].items()))
        if r["seam"]["strips"]:
            print(f"  seam lost: natural {r['seam']['natural_lost_db']:+.3f} dB, bright beside shadow "
                  f"{r['seam']['bright_beside_shadow_lost_db']:+.3f} dB ({r['seam']['strips']} strips)")
        print("  ripple: " + ", ".join(f"{k} {v:.2f}%" for k, v in r["ripple_percent"].items()))
    if args.report:
        settings = {"tile": TILE, "overlap": OVERLAP, "interior": INTERIOR, "floor": FLOOR, "cfa": args.cfa,
                    "seam_origins": [SEAM_A, SEAM_B, SEAM_NATURAL_REF, SEAM_SHADOW_REF], "seam_band": list(SEAM_BAND)}
        Path(args.report).write_text(json.dumps({
            "settings": settings,
            "sources": [{"path": s.path, "black": s.black, "white": s.white,
                         "display_wb": s.display_wb.tolist(), "display_gain": s.display_gain} for s in ev.sources],
            "tiles": [{"source": ev.sources[t.source].path, "y": t.y, "x": t.x} for t in ev.tiles],
            "results": results}, indent=2))
        print(f"\nreport: {args.report}")


if __name__ == "__main__":
    main()
