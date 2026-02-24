"""
Loss functions for X-Trans demosaicing.

Components:
- L1: pixel-level accuracy (drives PSNR)
- Gradient (Sobel): edge preservation
- MS-SSIM: multi-scale structural similarity (texture/detail)
- Chroma: penalizes false color artifacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    x = torch.arange(size).float() - size // 2
    kernel = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _gaussian_kernel_2d(size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create 2D Gaussian kernel for conv2d."""
    kernel_1d = _gaussian_kernel_1d(size, sigma)
    kernel_2d = kernel_1d.outer(kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, size, size).repeat(channels, 1, 1, 1)


class SobelGradientLoss(nn.Module):
    """Compare spatial gradients (edges) between prediction and target."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _sobel(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)
        gx = F.conv2d(x_flat, self.sobel_x, padding=1).reshape(B, C, H, W)
        gy = F.conv2d(x_flat, self.sobel_y, padding=1).reshape(B, C, H, W)
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gx_p, gy_p = self._sobel(pred)
        gx_t, gy_t = self._sobel(target)
        return ((gx_p - gx_t).abs() + (gy_p - gy_t).abs()).mean()


class ChromaLoss(nn.Module):
    """Penalize high-frequency chrominance (false color artifacts)."""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        sigma = kernel_size / 4.0
        kernel = _gaussian_kernel_2d(kernel_size, sigma, channels=1)
        self.register_buffer('lowpass', kernel[:1])  # Single channel

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # RGB to YCbCr (simplified)
        def to_chroma(rgb):
            r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
            cb = -0.169 * r - 0.331 * g + 0.500 * b
            cr = 0.500 * r - 0.419 * g - 0.081 * b
            return torch.cat([cb, cr], dim=1)

        pred_chroma = to_chroma(pred)
        target_chroma = to_chroma(target)

        # High-pass = original - low-pass
        pad = self.kernel_size // 2
        B, C, H, W = pred_chroma.shape
        pred_flat = pred_chroma.reshape(B * C, 1, H, W)
        target_flat = target_chroma.reshape(B * C, 1, H, W)

        pred_hp = pred_flat - F.conv2d(pred_flat, self.lowpass, padding=pad)
        target_hp = target_flat - F.conv2d(target_flat, self.lowpass, padding=pad)

        return F.l1_loss(pred_hp, target_hp)


class ZipperLoss(nn.Module):
    """Penalize spurious high-frequency oscillations (zipper artifacts).

    Uses the Laplacian (2nd-order derivative) to detect alternating pixel
    patterns that shouldn't exist in a properly demosaiced image.  The
    first-order Sobel gradient loss already penalizes edge errors, but
    zipper is specifically a *second-order* phenomenon — rapid sign
    alternation — that Sobel largely misses.
    """

    def __init__(self):
        super().__init__()
        laplacian = torch.tensor([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0],
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian', laplacian)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        pred_flat = pred.reshape(B * C, 1, H, W)
        target_flat = target.reshape(B * C, 1, H, W)
        lap_pred = F.conv2d(pred_flat, self.laplacian, padding=1)
        lap_target = F.conv2d(target_flat, self.laplacian, padding=1)
        return F.l1_loss(lap_pred, lap_target)


class ColorBiasLoss(nn.Module):
    """Penalize systematic color shift (DC bias) between prediction and target."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean(dim=(2, 3))    # (B, 3)
        target_mean = target.mean(dim=(2, 3))
        return F.l1_loss(pred_mean, target_mean)


class HighlightBiasLoss(nn.Module):
    """Penalize chrominance error in highlight regions (per-pixel).

    Isolates tint from brightness using log-space chrominance: for each
    pixel, subtract the mean log-channel value.  This is invariant to
    luminance and directly captures color-ratio errors (the "tint").
    A soft highlight mask focuses the loss on bright / near-clipped pixels
    where reconstruction artifacts are most visible.

    When clip_levels (per-channel clip level in WB'd space) is provided,
    the threshold is a fraction of clip level (e.g. 0.5 = ramp starts at
    50% of clip, reaches full weight at the clip point).
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                clip_levels: torch.Tensor | None = None) -> torch.Tensor:
        if clip_levels is not None:
            # clip_levels: (B, 3) → (B, 3, 1, 1)
            cl = clip_levels.unsqueeze(-1).unsqueeze(-1)
            # Per-channel ratio to clip level, take max across channels
            ratio = target / (cl + 1e-8)  # (B, 3, H, W)
            max_ratio = ratio.max(dim=1).values  # (B, H, W)
            # Ramp: 0 at threshold fraction, 1 at clip level
            weight = ((max_ratio - self.threshold) / (1.0 - self.threshold)).clamp(0, 1)
        else:
            # Fallback: absolute threshold (backward compat)
            tgt_max = target.max(dim=1).values  # (B, H, W)
            weight = ((tgt_max - self.threshold) / max(self.threshold, 1e-6)).clamp(0, 1)

        total_weight = weight.sum()
        if total_weight < 1.0:
            return pred.new_tensor(0.0)

        # Log-space chrominance: subtract per-pixel mean log-channel to
        # remove brightness and isolate color ratios.
        eps = 1e-4
        pred_log = torch.log(pred.clamp(min=eps))      # (B, 3, H, W)
        target_log = torch.log(target.clamp(min=eps))

        pred_chroma = pred_log - pred_log.mean(dim=1, keepdim=True)
        target_chroma = target_log - target_log.mean(dim=1, keepdim=True)

        weight = weight.unsqueeze(1)  # (B, 1, H, W)
        diff = (pred_chroma - target_chroma).abs() * weight

        return diff.sum() / (total_weight * 3)


class HighlightRelativeLoss(nn.Module):
    """Relative L1 loss for highlight regions.

    Computes |pred - target| / (target + eps), weighted by a soft highlight
    mask.  This normalizes errors by pixel brightness so the network gets
    proportional gradient signal for bright pixels that would otherwise be
    drowned out by the larger number of normal-brightness demosaic pixels.

    A 0.5 absolute error at target=4.0 contributes the same as a 0.05
    absolute error at target=0.4 — both are 12.5% relative error.

    The highlight mask ramps from 0 at ``threshold`` fraction of clip level
    to 1.0 at the clip level, so only bright/near-clipped pixels are
    affected.  Normal demosaic pixels are handled by the standard L1 term.
    """

    def __init__(self, threshold: float = 0.5, eps: float = 0.01):
        super().__init__()
        self.threshold = threshold
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                clip_levels: torch.Tensor | None = None) -> torch.Tensor:
        # Build soft highlight mask
        if clip_levels is not None:
            cl = clip_levels.unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1)
            ratio = target / (cl + 1e-8)  # (B, 3, H, W)
            max_ratio = ratio.max(dim=1).values  # (B, H, W)
            weight = ((max_ratio - self.threshold) / (1.0 - self.threshold)).clamp(0, 1)
        else:
            tgt_max = target.max(dim=1).values  # (B, H, W)
            weight = ((tgt_max - self.threshold) / max(self.threshold, 1e-6)).clamp(0, 1)

        total_weight = weight.sum()
        if total_weight < 1.0:
            return pred.new_tensor(0.0)

        # Relative L1: normalize error by target magnitude
        rel_error = (pred - target).abs() / (target.abs() + self.eps)

        weight = weight.unsqueeze(1)  # (B, 1, H, W) broadcast over channels
        weighted = rel_error * weight
        return weighted.sum() / (total_weight * 3)


class HighlightGradientLoss(nn.Module):
    """Direction-aware gradient (Sobel) loss weighted by highlight mask.

    Compares horizontal and vertical Sobel derivatives separately so the
    model is penalised for both wrong magnitude *and* wrong direction.
    Weighted by the same soft highlight ramp used by the other HL losses.
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        sobel_x = torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _sobel(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (gx, gy) each shaped (B, C, H, W)."""
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)
        gx = F.conv2d(x_flat, self.sobel_x, padding=1).reshape(B, C, H, W)
        gy = F.conv2d(x_flat, self.sobel_y, padding=1).reshape(B, C, H, W)
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                clip_levels: torch.Tensor | None = None) -> torch.Tensor:
        # Build soft highlight mask from target brightness
        if clip_levels is not None:
            cl = clip_levels.unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1)
            ratio = target / (cl + 1e-8)  # (B, 3, H, W)
            max_ratio = ratio.max(dim=1).values  # (B, H, W)
            weight = ((max_ratio - self.threshold) / (1.0 - self.threshold)).clamp(0, 1)
        else:
            tgt_max = target.max(dim=1).values
            weight = ((tgt_max - self.threshold) / max(self.threshold, 1e-6)).clamp(0, 1)

        total_weight = weight.sum()
        if total_weight < 1.0:
            return pred.new_tensor(0.0)

        gx_pred, gy_pred = self._sobel(pred)
        gx_target, gy_target = self._sobel(target)

        diff = (gx_pred - gx_target).abs() + (gy_pred - gy_target).abs()  # (B, C, H, W)
        weight = weight.unsqueeze(1)  # (B, 1, H, W)
        return (diff * weight).sum() / (total_weight * 3)


class SSIM(nn.Module):
    """Single-scale Structural Similarity Index."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5, channels: int = 3,
                 data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.data_range = data_range
        self.register_buffer('kernel', _gaussian_kernel_2d(window_size, sigma, channels))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns SSIM value (higher is better, max 1.0)."""
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2
        pad = self.window_size // 2

        mu1 = F.conv2d(pred, self.kernel, padding=pad, groups=self.channels)
        mu2 = F.conv2d(target, self.kernel, padding=pad, groups=self.channels)

        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, self.kernel, padding=pad, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.kernel, padding=pad, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.kernel, padding=pad, groups=self.channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


class MSSSIM(nn.Module):
    """
    Multi-Scale Structural Similarity Index.
    
    Computes SSIM at multiple scales (via downsampling) and combines them.
    Better captures structure at different frequencies than single-scale SSIM.
    
    Default weights from Wang et al. 2003 (5 scales).
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 3,
        weights: list[float] | None = None,
        data_range: float = 1.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.data_range = data_range
        # Default weights for 5 scales (from the MS-SSIM paper)
        self.weights = weights or [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.n_scales = len(self.weights)
        self.register_buffer('kernel', _gaussian_kernel_2d(window_size, sigma, channels))

    def _ssim_components(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute luminance*contrast (l*c) and structure (s) components."""
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2
        C3 = C2 / 2
        pad = self.window_size // 2

        mu1 = F.conv2d(pred, self.kernel, padding=pad, groups=self.channels)
        mu2 = F.conv2d(target, self.kernel, padding=pad, groups=self.channels)

        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, self.kernel, padding=pad, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.kernel, padding=pad, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.kernel, padding=pad, groups=self.channels) - mu1_mu2

        # Clamp variances to avoid sqrt of negative
        sigma1_sq = torch.clamp(sigma1_sq, min=0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0)
        sigma1 = torch.sqrt(sigma1_sq)
        sigma2 = torch.sqrt(sigma2_sq)

        # Luminance comparison
        l = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        # Contrast-structure (combined for numerical stability at coarse scales)
        cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

        return l.mean(), cs.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns MS-SSIM value (higher is better, max 1.0)."""
        weights = torch.tensor(self.weights, device=pred.device, dtype=pred.dtype)
        
        msssim = torch.ones(1, device=pred.device, dtype=pred.dtype)
        
        for i in range(self.n_scales):
            if i > 0:
                # Downsample by 2x
                pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
                target = F.avg_pool2d(target, kernel_size=2, stride=2)
            
            # Check minimum size
            if pred.shape[2] < self.window_size or pred.shape[3] < self.window_size:
                # Not enough resolution for this scale, use remaining weight on last valid
                break
            
            l, cs = self._ssim_components(pred, target)
            
            if i == self.n_scales - 1:
                # Last scale: include luminance
                msssim = msssim * (l.clamp(min=1e-8) ** weights[i]) * (cs.clamp(min=1e-8) ** weights[i])
            else:
                # Intermediate scales: only contrast-structure
                msssim = msssim * (cs.clamp(min=1e-8) ** weights[i])
        
        return msssim


class DemosaicLoss(nn.Module):
    """
    Unified loss for X-Trans demosaicing training.
    
    Components:
    - L1: pixel accuracy (PSNR)
    - MS-SSIM: multi-scale structure (texture/detail)
    - Gradient: edge preservation  
    - Chroma: false color penalty
    Presets:
    - "base": L1-heavy for initial training (high PSNR)
    - "finetune": MS-SSIM + gradient for texture recovery
    
    Options:
    - per_channel_norm: normalize loss per channel before combining (addresses G >> R,B)
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        msssim_weight: float = 0.0,
        gradient_weight: float = 0.1,
        chroma_weight: float = 0.05,
        color_bias_weight: float = 0.0,
        zipper_weight: float = 0.0,
        hl_bias_weight: float = 0.0,
        hl_rel_weight: float = 0.0,
        hl_grad_weight: float = 0.0,
        hl_threshold: float = 0.5,
        hl_fade: float = 0.0,
        per_channel_norm: bool = False,
        use_huber: bool = False,
        huber_delta: float = 1.0,
        data_range: float = 1.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.msssim_weight = msssim_weight
        self.gradient_weight = gradient_weight
        self.chroma_weight = chroma_weight
        self.color_bias_weight = color_bias_weight
        self.zipper_weight = zipper_weight
        self.hl_bias_weight = hl_bias_weight
        self.hl_rel_weight = hl_rel_weight
        self.hl_grad_weight = hl_grad_weight
        self.hl_threshold = hl_threshold
        self.hl_fade = hl_fade
        self.per_channel_norm = per_channel_norm
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.data_range = data_range

        self.msssim = MSSSIM(data_range=data_range) if msssim_weight > 0 else None
        self.gradient = SobelGradientLoss() if gradient_weight > 0 else None
        self.chroma = ChromaLoss() if chroma_weight > 0 else None
        self.color_bias = ColorBiasLoss() if color_bias_weight > 0 else None
        self.zipper = ZipperLoss() if zipper_weight > 0 else None
        self.hl_bias = HighlightBiasLoss(threshold=hl_threshold) if hl_bias_weight > 0 else None
        self.hl_rel = HighlightRelativeLoss(threshold=hl_threshold) if hl_rel_weight > 0 else None
        self.hl_grad = HighlightGradientLoss(threshold=hl_threshold) if hl_grad_weight > 0 else None

    @classmethod
    def base(cls, data_range: float = 1.0) -> "DemosaicLoss":
        """Preset for initial training: L1-focused for high PSNR."""
        return cls(l1_weight=1.0, msssim_weight=0.0, gradient_weight=0.1, chroma_weight=0.05,
                   zipper_weight=0.05, data_range=data_range)

    @classmethod
    def finetune(cls, msssim_weight: float = 0.3, gradient_weight: float = 0.2,
                 data_range: float = 1.0) -> "DemosaicLoss":
        """Preset for fine-tuning: MS-SSIM + gradient for texture."""
        return cls(
            l1_weight=0.5,
            msssim_weight=msssim_weight,
            gradient_weight=gradient_weight,
            chroma_weight=0.02,
            zipper_weight=0.1,
            data_range=data_range,
        )

    def _hl_mask(self, target: torch.Tensor,
                 clip_levels: torch.Tensor | None) -> torch.Tensor | None:
        """Build soft highlight mask (B, 1, H, W). Returns None if not needed."""
        if clip_levels is not None:
            cl = clip_levels.unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1)
            ratio = target / (cl + 1e-8)
            max_ratio = ratio.max(dim=1, keepdim=True).values  # (B, 1, H, W)
        else:
            max_ratio = target.max(dim=1, keepdim=True).values
        return ((max_ratio - self.hl_threshold) / (1.0 - self.hl_threshold)).clamp(0, 1)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
        clip_levels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        components = {}
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Precompute highlight mask if any hl feature is active
        hl_mask = None
        need_hl_mask = (self.hl_fade > 0
                        or (self.hl_bias is not None and self.hl_bias_weight > 0)
                        or (self.hl_rel is not None and self.hl_rel_weight > 0)
                        or (self.hl_grad is not None and self.hl_grad_weight > 0))
        if need_hl_mask:
            hl_mask = self._hl_mask(target, clip_levels)

        # L1 or Huber (optionally per-channel normalized)
        if self.l1_weight > 0:
            loss_name = 'huber' if self.use_huber else 'l1'

            if self.hl_fade > 0 and hl_mask is not None:
                # Per-pixel loss with highlight fade
                if self.use_huber:
                    px = F.huber_loss(pred, target, delta=self.huber_delta, reduction='none')
                else:
                    px = (pred - target).abs()
                # Fade out pixel loss in highlight regions
                fade_weight = 1.0 - self.hl_fade * hl_mask  # (B, 1, H, W)
                px = px * fade_weight

                if self.per_channel_norm:
                    loss_r = px[:, 0].mean()
                    loss_g = px[:, 1].mean()
                    loss_b = px[:, 2].mean()
                    pixel_loss = (loss_r + loss_g + loss_b) / 3
                    components[f'{loss_name}_r'] = loss_r.item()
                    components[f'{loss_name}_g'] = loss_g.item()
                    components[f'{loss_name}_b'] = loss_b.item()
                else:
                    pixel_loss = px.mean()
            else:
                loss_fn = (lambda p, t: F.huber_loss(p, t, delta=self.huber_delta)) if self.use_huber else F.l1_loss
                if self.per_channel_norm:
                    loss_r = loss_fn(pred[:, 0], target[:, 0])
                    loss_g = loss_fn(pred[:, 1], target[:, 1])
                    loss_b = loss_fn(pred[:, 2], target[:, 2])
                    pixel_loss = (loss_r + loss_g + loss_b) / 3
                    components[f'{loss_name}_r'] = loss_r.item()
                    components[f'{loss_name}_g'] = loss_g.item()
                    components[f'{loss_name}_b'] = loss_b.item()
                else:
                    pixel_loss = loss_fn(pred, target)
            components[loss_name] = pixel_loss.item()
            total = total + self.l1_weight * pixel_loss

        # MS-SSIM (1 - msssim, so lower is better)
        if self.msssim is not None and self.msssim_weight > 0:
            msssim_val = self.msssim(pred, target)
            msssim_loss = 1 - msssim_val
            components['msssim'] = msssim_val.item()
            total = total + self.msssim_weight * msssim_loss

        # Gradient
        if self.gradient is not None and self.gradient_weight > 0:
            grad = self.gradient(pred, target)
            components['gradient'] = grad.item()
            total = total + self.gradient_weight * grad

        # Chroma
        if self.chroma is not None and self.chroma_weight > 0:
            chroma = self.chroma(pred, target)
            components['chroma'] = chroma.item()
            total = total + self.chroma_weight * chroma

        # Zipper (2nd-order oscillation penalty)
        if self.zipper is not None and self.zipper_weight > 0:
            zipper = self.zipper(pred, target)
            components['zipper'] = zipper.item()
            total = total + self.zipper_weight * zipper

        # Color bias (DC shift penalty)
        if self.color_bias is not None and self.color_bias_weight > 0:
            cb = self.color_bias(pred, target)
            components['color_bias'] = cb.item()
            total = total + self.color_bias_weight * cb

        # Highlight bias (DC shift in bright regions only)
        if self.hl_bias is not None and self.hl_bias_weight > 0:
            hlb = self.hl_bias(pred, target, clip_levels=clip_levels)
            components['hl_bias'] = hlb.item()
            total = total + self.hl_bias_weight * hlb

        # Highlight relative L1 (proportional error in bright regions)
        if self.hl_rel is not None and self.hl_rel_weight > 0:
            hlr = self.hl_rel(pred, target, clip_levels=clip_levels)
            components['hl_rel'] = hlr.item()
            total = total + self.hl_rel_weight * hlr

        # Highlight gradient (Sobel edge preservation in bright regions)
        if self.hl_grad is not None and self.hl_grad_weight > 0:
            hlg = self.hl_grad(pred, target, clip_levels=clip_levels)
            components['hl_grad'] = hlg.item()
            total = total + self.hl_grad_weight * hlg

        components['total'] = total.item()
        return total, components


# Backwards compatibility aliases
CombinedLoss = DemosaicLoss
