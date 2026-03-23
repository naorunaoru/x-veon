# SPDX-License-Identifier: MIT
# Copyright (c) 2024-present X-Veon contributors
"""
Loss functions for X-Trans demosaicing.

Components:
- L1: pixel-level accuracy (drives PSNR)
- Gradient (Sobel): edge preservation
- MS-SSIM: multi-scale structural similarity (texture/detail)
- Chroma: penalizes false color artifacts
- FFT: frequency-domain magnitude spectrum (periodic artifact penalty)
- Local variance: texture consistency (penalizes mushy/averaged-out detail)
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
    """Penalize false-color artifacts using a guided filter.

    A guided filter (He et al. 2013) with the luminance channel as guide
    produces an edge-preserving lowpass of the chroma channels.  The highpass
    residual (original minus guided lowpass) then contains only chroma
    variations that are NOT explained by luminance edges — i.e. false color.
    """

    def __init__(self, radius: int = 2, eps: float = 1e-2):
        super().__init__()
        self.radius = radius
        self.eps = eps

    @staticmethod
    def _box_mean(x: torch.Tensor, r: int) -> torch.Tensor:
        """Box (mean) filter with reflect-padding for correct borders."""
        return F.avg_pool2d(
            F.pad(x, [r, r, r, r], mode='reflect'),
            kernel_size=2 * r + 1, stride=1, padding=0,
        )

    def _guided_highpass(
        self, luma: torch.Tensor, chroma: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-preserving highpass: chroma minus guided-filter lowpass.

        luma:   (B, 1, H, W)  — guide signal
        chroma: (B, 2, H, W)  — Cb, Cr channels to filter
        """
        r, eps, bm = self.radius, self.eps, self._box_mean
        mean_I = bm(luma, r)
        mean_p = bm(chroma, r)
        cov_Ip = bm(luma * chroma, r) - mean_I * mean_p
        var_I = bm(luma * luma, r) - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        lowpass = bm(a, r) * luma + bm(b, r)
        return chroma - lowpass

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def rgb_to_y_chroma(rgb):
            r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
            y  =  0.299 * r + 0.587 * g + 0.114 * b
            cb = -0.169 * r - 0.331 * g + 0.500 * b
            cr =  0.500 * r - 0.419 * g - 0.081 * b
            return y, torch.cat([cb, cr], dim=1)

        pred_y, pred_chroma = rgb_to_y_chroma(pred)
        target_y, target_chroma = rgb_to_y_chroma(target)

        pred_hp = self._guided_highpass(pred_y, pred_chroma)
        target_hp = self._guided_highpass(target_y, target_chroma)

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


class LocalVarianceLoss(nn.Module):
    """Penalize differences in local texture energy between prediction and target.

    Computes variance in sliding windows: var = E[x²] - E[x]².
    L1-trained networks tend to produce lower local variance (mushy textures)
    because averaging minimizes L1/L2 but kills fine detail.  This loss
    directly penalizes that variance gap — cheaper than a VGG forward pass.
    """

    def __init__(self, window_size: int = 7):
        super().__init__()
        self.window_size = window_size

    def _local_variance(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, C, H, W) local variance map."""
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)
        r = self.window_size // 2
        padded = F.pad(x_flat, [r, r, r, r], mode='reflect')
        mean = F.avg_pool2d(padded, self.window_size, stride=1)
        sq_mean = F.avg_pool2d(padded ** 2, self.window_size, stride=1)
        return (sq_mean - mean ** 2).clamp(min=0).reshape(B, C, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._local_variance(pred), self._local_variance(target))


class FFTLoss(nn.Module):
    """L1 loss on the magnitude spectrum of pred vs target.

    Penalizes periodic artifacts (e.g. grid patterns from demosaicing)
    that spatial-domain losses are blind to.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # cuFFT requires float32 for non-power-of-2 sizes under AMP
        diff = torch.fft.rfft2(pred.float()) - torch.fft.rfft2(target.float())
        norm = (pred.shape[-2] * pred.shape[-1]) ** 0.5
        return diff.abs().mean() / norm


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
    - FFT: frequency-domain magnitude spectrum (periodic artifacts)
    - Texture: local variance consistency (anti-mush)
    Presets:
    - "base": L1-heavy for initial training (high PSNR)
    - "finetune": MS-SSIM + gradient for texture recovery
    
    Options:
    - per_channel_norm: normalize loss per channel before combining (addresses G >> R,B)
    - recon_only: compute L1/Huber only on pixels under reconstruction (not sampled by CFA),
      with a small known_pixel_weight penalty to prevent drift at sampled positions
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        msssim_weight: float = 0.0,
        gradient_weight: float = 0.1,
        chroma_weight: float = 0.05,
        color_bias_weight: float = 0.0,
        zipper_weight: float = 0.0,
        fft_weight: float = 0.0,
        texture_weight: float = 0.0,
        texture_window: int = 7,
        per_channel_norm: bool = False,
        use_huber: bool = False,
        huber_delta: float = 1.0,
        data_range: float = 1.0,
        recon_only: bool = False,
        known_pixel_weight: float = 0.1,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.msssim_weight = msssim_weight
        self.gradient_weight = gradient_weight
        self.chroma_weight = chroma_weight
        self.color_bias_weight = color_bias_weight
        self.zipper_weight = zipper_weight
        self.fft_weight = fft_weight
        self.texture_weight = texture_weight
        self.per_channel_norm = per_channel_norm
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.data_range = data_range
        self.recon_only = recon_only
        self.known_pixel_weight = known_pixel_weight

        self.msssim = MSSSIM(data_range=data_range) if msssim_weight > 0 else None
        self.gradient = SobelGradientLoss() if gradient_weight > 0 else None
        self.chroma = ChromaLoss() if chroma_weight > 0 else None
        self.color_bias = ColorBiasLoss() if color_bias_weight > 0 else None
        self.zipper = ZipperLoss() if zipper_weight > 0 else None
        self.fft = FFTLoss() if fft_weight > 0 else None
        self.texture = LocalVarianceLoss(window_size=texture_window) if texture_weight > 0 else None

    @classmethod
    def base(cls, data_range: float = 1.0) -> "DemosaicLoss":
        """Preset for initial training: L1-focused for high PSNR."""
        return cls(l1_weight=1.0, msssim_weight=0.0, gradient_weight=0.1, chroma_weight=0.05,
                   zipper_weight=0.05, data_range=data_range)

    @classmethod
    def finetune(cls, msssim_weight: float = 0.3, gradient_weight: float = 0.2,
                 data_range: float = 1.0) -> "DemosaicLoss":
        """Preset for fine-tuning: MS-SSIM + gradient + texture for detail recovery."""
        return cls(
            l1_weight=0.5,
            msssim_weight=msssim_weight,
            gradient_weight=gradient_weight,
            chroma_weight=0.02,
            zipper_weight=0.1,
            texture_weight=0.1,
            data_range=data_range,
        )

    def _masked_loss(
        self, pred: torch.Tensor, target: torch.Tensor,
        mask: torch.Tensor, loss_fn,
    ) -> torch.Tensor:
        """Compute mean loss over masked pixels only."""
        # mask may be (1, C, H, W) broadcasting over batch — scale denominator
        # to account for the batch dimension in the numerator's .sum()
        B = pred.shape[0]
        denom = mask.sum().clamp(min=1) * B
        diff = (pred - target).abs() if loss_fn is F.l1_loss else None
        if diff is not None:
            return (diff * mask).sum() / denom
        # Huber: element-wise then mask
        elem = F.huber_loss(pred, target, delta=self.huber_delta, reduction='none')
        return (elem * mask).sum() / denom

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
        channel_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components: dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Build known/unknown masks for reconstruction-only mode
        # channel_masks: (B, 3, H, W) binary — 1 where CFA samples that channel
        use_recon_mask = self.recon_only and channel_masks is not None
        if use_recon_mask:
            known_mask = channel_masks  # (B, 3, H, W)
            unknown_mask = 1.0 - known_mask

        # L1 or Huber (optionally per-channel normalized)
        if self.l1_weight > 0:
            loss_name = 'huber' if self.use_huber else 'l1'
            loss_fn = (lambda p, t: F.huber_loss(p, t, delta=self.huber_delta)) if self.use_huber else F.l1_loss
            if use_recon_mask:
                # Loss on reconstructed (unknown) pixels
                recon_loss = self._masked_loss(pred, target, unknown_mask, loss_fn)
                # Small penalty to preserve known pixels
                known_loss = self._masked_loss(pred, target, known_mask, loss_fn)
                pixel_loss = recon_loss + self.known_pixel_weight * known_loss
                components[f'{loss_name}_recon'] = recon_loss.detach()
                components[f'{loss_name}_known'] = known_loss.detach()
            elif self.per_channel_norm:
                loss_r = loss_fn(pred[:, 0], target[:, 0])
                loss_g = loss_fn(pred[:, 1], target[:, 1])
                loss_b = loss_fn(pred[:, 2], target[:, 2])
                pixel_loss = (loss_r + loss_g + loss_b) / 3
                components[f'{loss_name}_r'] = loss_r.detach()
                components[f'{loss_name}_g'] = loss_g.detach()
                components[f'{loss_name}_b'] = loss_b.detach()
            else:
                pixel_loss = loss_fn(pred, target)
            components[loss_name] = pixel_loss.detach()
            total = total + self.l1_weight * pixel_loss

        # MS-SSIM (1 - msssim, so lower is better)
        if self.msssim is not None and self.msssim_weight > 0:
            msssim_val = self.msssim(pred, target)
            msssim_loss = 1 - msssim_val
            components['msssim'] = msssim_val.detach()
            total = total + self.msssim_weight * msssim_loss

        # Gradient
        if self.gradient is not None and self.gradient_weight > 0:
            grad = self.gradient(pred, target)
            components['gradient'] = grad.detach()
            total = total + self.gradient_weight * grad

        # Chroma
        if self.chroma is not None and self.chroma_weight > 0:
            chroma = self.chroma(pred, target)
            components['chroma'] = chroma.detach()
            total = total + self.chroma_weight * chroma

        # Zipper (2nd-order oscillation penalty)
        if self.zipper is not None and self.zipper_weight > 0:
            zipper = self.zipper(pred, target)
            components['zipper'] = zipper.detach()
            total = total + self.zipper_weight * zipper

        # FFT (frequency-domain magnitude spectrum)
        if self.fft is not None and self.fft_weight > 0:
            fft = self.fft(pred, target)
            components['fft'] = fft.detach()
            total = total + self.fft_weight * fft

        # Texture (local variance consistency)
        if self.texture is not None and self.texture_weight > 0:
            tex = self.texture(pred, target)
            components['texture'] = tex.detach()
            total = total + self.texture_weight * tex

        # Color bias (DC shift penalty)
        if self.color_bias is not None and self.color_bias_weight > 0:
            cb = self.color_bias(pred, target)
            components['color_bias'] = cb.detach()
            total = total + self.color_bias_weight * cb

        components['total'] = total.detach()
        return total, components


# Backwards compatibility aliases
CombinedLoss = DemosaicLoss
