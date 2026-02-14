"""
Expanded torture test dataset v2 with 4x supersampling, smart colors, and fractals.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
import random
import colorsys

from cfa import make_cfa_mask, make_channel_masks, CFA_REGISTRY


def mosaic_linear(rgb: torch.Tensor, cfa: torch.Tensor) -> torch.Tensor:
    """Apply CFA mosaic to RGB image."""
    h, w = cfa.shape
    out = torch.zeros(1, h, w)
    for c in range(3):
        out[0] += rgb[c] * (cfa == c).float()
    return out


def rotate_coords(xx: torch.Tensor, yy: torch.Tensor, angle: float, cx: float, cy: float):
    """Rotate coordinate grids around center."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    xx_c = xx - cx
    yy_c = yy - cy
    xx_rot = xx_c * cos_a - yy_c * sin_a + cx
    yy_rot = xx_c * sin_a + yy_c * cos_a + cy
    return xx_rot, yy_rot


def hsv_to_linear_rgb(h: float, s: float, v: float) -> torch.Tensor:
    """Convert HSV to linear RGB."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    def to_linear(c):
        return c ** 2.2
    return torch.tensor([to_linear(r), to_linear(g), to_linear(b)])


def rgb_contrast(c1: torch.Tensor, c2: torch.Tensor) -> float:
    """Simple Euclidean contrast in linear RGB."""
    return torch.sqrt(((c1 - c2) ** 2).sum()).item()


# ============ Fractal generators ============

def find_julia_boundary_point(c_real: float, c_imag: float, rng: random.Random,
                               samples: int = 500, max_iter: int = 100) -> tuple:
    """Find a point near the Julia set boundary (interesting zoom target)."""
    best_point = (0.0, 0.0)
    best_score = 0
    
    for _ in range(samples):
        x = rng.uniform(-1.5, 1.5)
        y = rng.uniform(-1.5, 1.5)
        z = complex(x, y)
        c = complex(c_real, c_imag)
        
        for i in range(max_iter):
            if abs(z) > 2:
                break
            z = z * z + c
        
        # Points escaping mid-way are at boundary
        score = min(i, max_iter - i)
        if score > best_score:
            best_score = score
            best_point = (x, y)
    
    return best_point


def julia_set(h: int, w: int, c_real: float, c_imag: float, 
              max_iter: int = 100, zoom: float = 1.0, 
              center: tuple = (0.0, 0.0)) -> torch.Tensor:
    """Generate Julia set fractal with zoom support."""
    extent = 1.5 / zoom
    cx, cy = center
    
    x = torch.linspace(cx - extent, cx + extent, w)
    y = torch.linspace(cy - extent, cy + extent, h)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    Z = X + 1j * Y
    C = complex(c_real, c_imag)
    
    result = torch.zeros(h, w)
    for i in range(max_iter):
        mask = torch.abs(Z) <= 2
        Z = torch.where(mask, Z * Z + C, Z)
        result += mask.float()
    
    return result / max_iter


def perlin_octave(h: int, w: int, freq: float, seed: int = 0) -> torch.Tensor:
    """Single octave of Perlin-like noise."""
    rng = torch.Generator().manual_seed(seed)
    
    grid_h = int(h * freq) + 2
    grid_w = int(w * freq) + 2
    angles = torch.rand(grid_h, grid_w, generator=rng) * 2 * math.pi
    grad_x = torch.cos(angles)
    grad_y = torch.sin(angles)
    
    y_coords = torch.linspace(0, (grid_h - 1) * 0.99, h)
    x_coords = torch.linspace(0, (grid_w - 1) * 0.99, w)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")
    
    x0 = X.floor().long()
    y0 = Y.floor().long()
    x1 = x0 + 1
    y1 = y0 + 1
    
    dx = X - x0.float()
    dy = Y - y0.float()
    
    def smoothstep(t):
        return t * t * (3 - 2 * t)
    
    sx = smoothstep(dx)
    sy = smoothstep(dy)
    
    def dot_grid(gx, gy, dx, dy):
        return gx * dx + gy * dy
    
    n00 = dot_grid(grad_x[y0, x0], grad_y[y0, x0], dx, dy)
    n10 = dot_grid(grad_x[y0, x1], grad_y[y0, x1], dx - 1, dy)
    n01 = dot_grid(grad_x[y1, x0], grad_y[y1, x0], dx, dy - 1)
    n11 = dot_grid(grad_x[y1, x1], grad_y[y1, x1], dx - 1, dy - 1)
    
    nx0 = n00 * (1 - sx) + n10 * sx
    nx1 = n01 * (1 - sx) + n11 * sx
    result = nx0 * (1 - sy) + nx1 * sy
    
    return result


def fractal_noise(h: int, w: int, octaves: int = 5, persistence: float = 0.5,
                  base_freq: float = 0.02, seed: int = 0) -> torch.Tensor:
    """Fractal Brownian motion - multiple octaves of Perlin noise."""
    result = torch.zeros(h, w)
    amplitude = 1.0
    freq = base_freq
    max_val = 0.0
    
    for i in range(octaves):
        result += perlin_octave(h, w, freq, seed + i * 1000) * amplitude
        max_val += amplitude
        amplitude *= persistence
        freq *= 2
    
    result = (result / max_val + 1) / 2
    return result.clamp(0, 1)


# ============ Color picker ============

class ColorPicker:
    """Smart color selection with multiple strategies."""
    
    COMPLEMENTARY = [
        ([0.35, 0.05, 0.05], [0.05, 0.35, 0.35]),
        ([0.05, 0.35, 0.05], [0.35, 0.05, 0.35]),
        ([0.05, 0.05, 0.35], [0.35, 0.35, 0.05]),
    ]
    
    CROSS_CHANNEL = [
        ([0.4, 0.05, 0.05], [0.05, 0.4, 0.05]),
        ([0.05, 0.4, 0.05], [0.05, 0.05, 0.4]),
        ([0.05, 0.05, 0.4], [0.4, 0.05, 0.05]),
    ]
    
    NEAR_NEUTRAL = [
        ([0.20, 0.18, 0.18], [0.18, 0.20, 0.18]),
        ([0.25, 0.25, 0.22], [0.22, 0.22, 0.25]),
        ([0.30, 0.28, 0.28], [0.15, 0.16, 0.16]),
    ]
    
    def __init__(self, rng: random.Random):
        self.rng = rng
    
    def _hsv_random(self, min_v=0.1, max_v=0.5, min_s=0.3, max_s=1.0) -> torch.Tensor:
        h = self.rng.random()
        s = self.rng.uniform(min_s, max_s)
        v = self.rng.uniform(min_v, max_v)
        return hsv_to_linear_rgb(h, s, v)
    
    def _hsv_pair_contrasting(self, min_contrast=0.15) -> tuple:
        for _ in range(20):
            c1 = self._hsv_random()
            c2 = self._hsv_random()
            if rgb_contrast(c1, c2) >= min_contrast:
                return c1, c2
        h1 = self.rng.random()
        h2 = (h1 + 0.5) % 1.0
        v = self.rng.uniform(0.15, 0.45)
        return hsv_to_linear_rgb(h1, 0.7, v), hsv_to_linear_rgb(h2, 0.7, v)
    
    def _from_list(self, pairs: list) -> tuple:
        c1, c2 = self.rng.choice(pairs)
        def vary(c):
            return torch.tensor([max(0.02, min(0.6, v + self.rng.uniform(-0.03, 0.03))) for v in c])
        return vary(c1), vary(c2)
    
    def get_pair(self, strategy: str = "mixed") -> tuple:
        if strategy == "mixed":
            r = self.rng.random()
            if r < 0.4:
                strategy = "hsv"
            elif r < 0.6:
                strategy = "complementary"
            elif r < 0.8:
                strategy = "cross_channel"
            else:
                strategy = "near_neutral"
        
        if strategy == "hsv":
            return self._hsv_pair_contrasting()
        elif strategy == "complementary":
            return self._from_list(self.COMPLEMENTARY)
        elif strategy == "cross_channel":
            return self._from_list(self.CROSS_CHANNEL)
        elif strategy == "near_neutral":
            return self._from_list(self.NEAR_NEUTRAL)
        else:
            return self._hsv_pair_contrasting()


# ============ Main dataset ============

class TortureDatasetV2(Dataset):
    """
    Expanded synthetic torture test patterns with 4x supersampling.
    
    13 pattern types with smart colors and fractal support.
    """
    
    NUM_TYPES = 13
    SUPERSAMPLE = 4
    
    JULIA_PARAMS = [
        (-0.7, 0.27015),
        (-0.8, 0.156),
        (0.285, 0.01),
        (-0.4, 0.6),
        (0.355, 0.355),
        (-0.54, 0.54),
        (0.37, 0.1),
        (-0.123, 0.745),
    ]
    
    def __init__(self, size: int = 288, num_patterns: int = 500,
                 add_noise: bool = True, noise_prob: float = 0.5,
                 cfa_type: str = "xtrans"):
        self.size = size
        self.size_hi = size * self.SUPERSAMPLE
        self.num_patterns = num_patterns
        self.add_noise = add_noise
        self.noise_prob = noise_prob
        pattern = CFA_REGISTRY[cfa_type]
        self.cfa = make_cfa_mask(size, size, pattern)
        self.masks = make_channel_masks(size, size, pattern)
        
    def __len__(self):
        return self.num_patterns
    
    def _make_coords(self, hi_res: bool = True):
        s = self.size_hi if hi_res else self.size
        yy, xx = torch.meshgrid(
            torch.arange(s, dtype=torch.float32),
            torch.arange(s, dtype=torch.float32),
            indexing="ij"
        )
        return xx, yy
    
    def _downsample(self, rgb: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            rgb.unsqueeze(0), 
            size=(self.size, self.size), 
            mode="area"
        ).squeeze(0)
    
    def _apply_noise(self, rgb: torch.Tensor, rng) -> torch.Tensor:
        if not self.add_noise or rng.random() > self.noise_prob:
            return rgb
        
        read_sigma = rng.uniform(0.005, 0.02)
        shot_sigma = rng.uniform(0.01, 0.05)
        
        noise_std = torch.sqrt(shot_sigma * rgb.clamp(min=0) + read_sigma**2)
        noise = torch.randn_like(rgb) * noise_std
        return (rgb + noise).clamp(0, 1)
    
    def _fractal_to_rgb(self, fractal: torch.Tensor, rng, mode: str = "hue") -> torch.Tensor:
        """Convert fractal values to RGB."""
        h, w = fractal.shape
        rgb = torch.zeros(3, h, w)
        
        if mode == "hue":
            hue_offset = rng.random()
            hue = (fractal + hue_offset) % 1.0
            rgb[0] = (torch.cos(hue * 6.28) * 0.5 + 0.5) * 0.4 + 0.05
            rgb[1] = (torch.cos((hue - 0.33) * 6.28) * 0.5 + 0.5) * 0.4 + 0.05
            rgb[2] = (torch.cos((hue - 0.66) * 6.28) * 0.5 + 0.5) * 0.4 + 0.05
        elif mode == "gradient":
            c1 = hsv_to_linear_rgb(rng.random(), rng.uniform(0.5, 1.0), rng.uniform(0.2, 0.5))
            c2 = hsv_to_linear_rgb(rng.random(), rng.uniform(0.5, 1.0), rng.uniform(0.2, 0.5))
            for c in range(3):
                rgb[c] = c1[c] * (1 - fractal) + c2[c] * fractal
        else:  # bands
            bands = (fractal * rng.randint(4, 12)).floor() % 2
            c1 = hsv_to_linear_rgb(rng.random(), 0.8, 0.4)
            c2 = hsv_to_linear_rgb((rng.random() + 0.5) % 1, 0.8, 0.3)
            for c in range(3):
                rgb[c] = c1[c] * bands + c2[c] * (1 - bands)
        
        return rgb
    
    def __getitem__(self, idx):
        rng = random.Random(idx * 7919)
        pattern_type = idx % self.NUM_TYPES
        color_picker = ColorPicker(rng)
        
        h = w = self.size_hi
        cx, cy = w / 2, h / 2
        xx, yy = self._make_coords(hi_res=True)
        rgb = torch.zeros(3, h, w)
        
        angle = rng.uniform(0, 2 * math.pi)
        ss = self.SUPERSAMPLE
        
        if pattern_type == 0:
            # Diagonal stripes
            freq = rng.uniform(0.02, 0.4) / ss
            xx_r, yy_r = rotate_coords(xx, yy, angle, cx, cy)
            stripe = (torch.sin(xx_r * freq) > 0).float()
            c1, c2 = color_picker.get_pair("mixed")
            for c in range(3):
                rgb[c] = c1[c] * stripe + c2[c] * (1 - stripe)
                
        elif pattern_type == 1:
            # Parallel lines
            line_width = rng.randint(1, 12) * ss
            xx_r, yy_r = rotate_coords(xx, yy, angle, cx, cy)
            mask = ((xx_r / line_width).floor() % 2 == 0).float()
            c1, c2 = color_picker.get_pair("mixed")
            for c in range(3):
                rgb[c] = c1[c] * mask + c2[c] * (1 - mask)
                
        elif pattern_type == 2:
            # Checkerboard
            cell_size = rng.randint(2, 24) * ss
            xx_r, yy_r = rotate_coords(xx, yy, angle, cx, cy)
            checker = (((xx_r / cell_size).floor() + (yy_r / cell_size).floor()) % 2 == 0).float()
            c1, c2 = color_picker.get_pair("mixed")
            for c in range(3):
                rgb[c] = c1[c] * checker + c2[c] * (1 - checker)
                
        elif pattern_type == 3:
            # Concentric circles
            freq = rng.uniform(0.05, 0.5) / ss
            phase = rng.uniform(0, 2 * math.pi)
            dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
            ring = (torch.sin(dist * freq + phase) > 0).float()
            c1, c2 = color_picker.get_pair("mixed")
            for c in range(3):
                rgb[c] = c1[c] * ring + c2[c] * (1 - ring)
                
        elif pattern_type == 4:
            # Color gradient
            xx_r, yy_r = rotate_coords(xx, yy, angle, cx, cy)
            t = ((xx_r - xx_r.min()) / (xx_r.max() - xx_r.min() + 1e-6))
            c1, c2 = color_picker.get_pair("mixed")
            for c in range(3):
                rgb[c] = c1[c] * (1 - t) + c2[c] * t
                
        elif pattern_type == 5:
            # Siemens star
            num_spokes = rng.randint(8, 36)
            angle_map = torch.atan2(yy - cy, xx - cx)
            star = (torch.sin(angle_map * num_spokes) > 0).float()
            c1, c2 = color_picker.get_pair("mixed")
            for c in range(3):
                rgb[c] = c1[c] * star + c2[c] * (1 - star)
                
        elif pattern_type == 6:
            # Slanted edge
            xx_r, yy_r = rotate_coords(xx, yy, angle, cx, cy)
            edge = (xx_r > cx).float()
            c1, c2 = color_picker.get_pair("mixed")
            for c in range(3):
                rgb[c] = c1[c] * edge + c2[c] * (1 - edge)
                
        elif pattern_type == 7:
            # Nyquist grid
            grid_size = rng.randint(1, 2) * ss
            xx_r, yy_r = rotate_coords(xx, yy, angle, cx, cy)
            if rng.random() > 0.5:
                grid = (((xx_r / grid_size).floor() + (yy_r / grid_size).floor()) % 2 == 0).float()
            else:
                grid = ((xx_r / grid_size).floor() % 2 == 0).float()
            c1, c2 = color_picker.get_pair("mixed")
            for c in range(3):
                rgb[c] = c1[c] * grid + c2[c] * (1 - grid)
                
        elif pattern_type == 8:
            # Chromatic edges
            xx_r, yy_r = rotate_coords(xx, yy, angle, cx, cy)
            edge = (xx_r > cx).float()
            c1, c2 = color_picker.get_pair("complementary")
            for c in range(3):
                rgb[c] = c1[c] * (1 - edge) + c2[c] * edge
                
        elif pattern_type == 9:
            # Radial color wheel
            angle_map = torch.atan2(yy - cy, xx - cx)
            dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
            dist_norm = dist / dist.max()
            
            hue = (angle_map + math.pi) / (2 * math.pi)
            sat = rng.uniform(0.5, 1.0)
            val = rng.uniform(0.2, 0.5)
            
            rgb[0] = (torch.cos(hue * 6.28) * 0.5 + 0.5) * val * sat + 0.05
            rgb[1] = (torch.cos((hue - 0.33) * 6.28) * 0.5 + 0.5) * val * sat + 0.05
            rgb[2] = (torch.cos((hue - 0.66) * 6.28) * 0.5 + 0.5) * val * sat + 0.05
            rgb = rgb * (1 - dist_norm * 0.3)
                
        elif pattern_type == 10:
            # Brick pattern
            cell_h = rng.randint(4, 12) * ss
            cell_w = rng.randint(8, 24) * ss
            xx_r, yy_r = rotate_coords(xx, yy, angle, cx, cy)
            
            row = (yy_r / cell_h).floor()
            x_shifted = xx_r + (row % 2) * (cell_w / 2)
            col = (x_shifted / cell_w).floor()
            brick = ((row + col) % 2 == 0).float()
            
            c1, c2 = color_picker.get_pair("mixed")
            for c in range(3):
                rgb[c] = c1[c] * brick + c2[c] * (1 - brick)
                
        elif pattern_type == 11:
            # Julia set fractal with zoom
            params = self.JULIA_PARAMS[idx % len(self.JULIA_PARAMS)]
            c_real = params[0] + rng.uniform(-0.05, 0.05)
            c_imag = params[1] + rng.uniform(-0.05, 0.05)
            
            # 50% chance of zoomed view
            if rng.random() > 0.5:
                center = find_julia_boundary_point(c_real, c_imag, rng)
                zoom = rng.uniform(8, 40)
                max_iter = 120
            else:
                center = (0.0, 0.0)
                zoom = rng.uniform(0.8, 1.5)
                max_iter = 80
            
            fractal = julia_set(h, w, c_real, c_imag, max_iter=max_iter, 
                               zoom=zoom, center=center)
            color_mode = rng.choice(["hue", "gradient", "bands"])
            rgb = self._fractal_to_rgb(fractal, rng, color_mode)
            
        else:  # pattern_type == 12
            # Perlin noise fractal
            octaves = rng.randint(3, 6)
            persistence = rng.uniform(0.4, 0.7)
            base_freq = rng.uniform(0.01, 0.04)
            seed = rng.randint(0, 10000)
            
            fractal = fractal_noise(h, w, octaves, persistence, base_freq, seed)
            color_mode = rng.choice(["hue", "gradient", "bands"])
            rgb = self._fractal_to_rgb(fractal, rng, color_mode)
        
        # Downsample
        rgb = self._downsample(rgb)
        
        # Apply noise
        rgb = self._apply_noise(rgb, rng)
        rgb = rgb.clamp(0, 1)
        
        # Create CFA mosaic
        cfa_img = mosaic_linear(rgb, self.cfa)
        input_tensor = torch.cat([cfa_img, self.masks], dim=0)
        
        return input_tensor, rgb


def generate_dataset(output_dir: str, num_samples: int = 5000, size: int = 288):
    """Generate full dataset as .npy files for training."""
    import os
    import numpy as np
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    ds = TortureDatasetV2(size=size, num_patterns=num_samples, add_noise=True, noise_prob=0.5)
    
    print(f"Generating {num_samples} torture patterns to {output_dir}")
    print(f"Pattern types: {ds.NUM_TYPES}")
    
    for idx in tqdm(range(num_samples)):
        inp, target = ds[idx]
        
        # Save as compressed .npz (input CFA+masks and target RGB)
        np.savez_compressed(
            f"{output_dir}/{idx:05d}.npz",
            input=inp.numpy().astype(np.float16),
            target=target.numpy().astype(np.float16)
        )
    
    # Save metadata
    import json
    meta = {
        "num_samples": num_samples,
        "size": size,
        "num_types": ds.NUM_TYPES,
        "pattern_names": [
            "diagonal_stripes", "parallel_lines", "checkerboard", "concentric",
            "gradient", "siemens_star", "slanted_edge", "nyquist_grid",
            "chromatic_edge", "radial_color", "brick", "julia", "perlin"
        ],
        "supersample": ds.SUPERSAMPLE,
        "noise_prob": 0.5,
    }
    with open(f"{output_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Done! Saved {num_samples} samples + meta.json")


def generate_samples(output_dir: str = "/tmp/torture_v2", num_per_type: int = 2):
    """Generate sample PNG images for preview."""
    import os
    from PIL import Image
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    ds = TortureDatasetV2(size=288, num_patterns=1000, add_noise=False)
    
    names = [
        "diagonal_stripes", "parallel_lines", "checkerboard", "concentric",
        "gradient", "siemens_star", "slanted_edge", "nyquist_grid",
        "chromatic_edge", "radial_color", "brick", "julia", "perlin"
    ]
    
    for type_idx in range(ds.NUM_TYPES):
        for sample in range(num_per_type):
            idx = type_idx + sample * ds.NUM_TYPES
            inp, target = ds[idx]
            
            rgb = target.numpy().transpose(1, 2, 0)
            rgb = np.clip(rgb ** (1/2.2) * 255, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(rgb)
            path = f"{output_dir}/{names[type_idx]}_{sample}.png"
            img.save(path)
            print(f"Saved {path}")
    
    print(f"\nGenerated {ds.NUM_TYPES * num_per_type} samples in {output_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "dataset":
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/torture_dataset"
        num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
        generate_dataset(output_dir, num_samples)
    else:
        generate_samples()
