# SPDX-License-Identifier: MIT
# Copyright (c) 2024-present X-Veon contributors
"""
Loss and metric for demosaicing training.

Both work on encoded values: each patch is divided by its own target mean and passed
through a power curve. The loss therefore does not change when a patch is made brighter
or darker (while its target mean stays at or above MEAN_FLOOR), and every patch weighs
the same.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GAMMA = 1.0 / 2.2
TOE = 0.02          # gives the curve a finite slope at black
MEAN_FLOOR = 1e-4   # patches darker than this are scored on a fixed scale


def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    x = torch.arange(size).float() - size // 2
    kernel = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _gaussian_kernel_2d(size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create 2D Gaussian kernel for conv2d (used by the OLPF augmentation)."""
    kernel_1d = _gaussian_kernel_1d(size, sigma)
    kernel_2d = kernel_1d.outer(kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, size, size).repeat(channels, 1, 1, 1)


def encode(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Power curve on values / scale, continued as a straight line below zero.

    Written without a branch: the power is only ever taken of a non-negative number.
    (A torch.where between two branches evaluates the power for negative inputs too
    and returns NaN gradients there.)
    """
    u = values.float() / scale
    p = u.clamp(min=0.0)
    encoded: torch.Tensor = torch.pow(p + TOE, GAMMA) - TOE ** GAMMA + GAMMA * TOE ** (GAMMA - 1.0) * (u - p)
    return encoded


def encode_pair(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prediction and target with the target patch's own mean as the scale."""
    target = target.float()
    scale = target.mean(dim=(1, 2, 3), keepdim=True).clamp(min=MEAN_FLOOR)
    return encode(pred, scale), encode(target, scale)


class EncodedPSNR:
    """PSNR on encoded values, pooled over a whole pass.

    Squared error and element count are summed over every batch and the logarithm is
    taken once, so the score does not depend on how predictions are grouped in batches.
    """

    def __init__(self) -> None:
        self.sse: torch.Tensor | None = None
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        enc_pred, enc_target = encode_pair(pred.detach(), target)
        sse = ((enc_pred - enc_target) ** 2).sum()
        self.sse = sse if self.sse is None else self.sse + sse
        self.count += enc_pred.numel()

    def value(self) -> float:
        if self.sse is None or self.count == 0:
            return float("nan")
        mse = (self.sse / self.count).clamp(min=1e-10)
        return float(-10.0 * torch.log10(mse))


class ColorBiasLoss(nn.Module):
    """Penalize systematic color shift (DC bias) between prediction and target."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean(dim=(2, 3))    # (B, 3)
        target_mean = target.mean(dim=(2, 3))
        return F.l1_loss(pred_mean, target_mean)


class DemosaicLoss(nn.Module):
    """L1 (or Huber) between encoded prediction and encoded target.

    Options:
    - recon_only: score only the values the CFA did not sample, plus
      known_pixel_weight times the loss on the sampled ones.
    - color_bias_weight: mean colour shift penalty on un-encoded values (off by default).
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        color_bias_weight: float = 0.0,
        use_huber: bool = False,
        huber_delta: float = 1.0,
        recon_only: bool = False,
        known_pixel_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.color_bias_weight = color_bias_weight
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.recon_only = recon_only
        self.known_pixel_weight = known_pixel_weight
        self.color_bias = ColorBiasLoss() if color_bias_weight > 0 else None

    def _elementwise(self, enc_pred: torch.Tensor, enc_target: torch.Tensor) -> torch.Tensor:
        if self.use_huber:
            return F.huber_loss(enc_pred, enc_target, delta=self.huber_delta, reduction="none")
        return (enc_pred - enc_target).abs()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
        channel_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components: dict[str, torch.Tensor] = {}
        name = "huber" if self.use_huber else "l1"
        enc_pred, enc_target = encode_pair(pred, target)
        elem = self._elementwise(enc_pred, enc_target)

        if self.recon_only and channel_masks is not None:
            known = channel_masks.to(elem.dtype).expand_as(elem)
            unknown = 1.0 - known
            recon = (elem * unknown).sum() / unknown.sum().clamp(min=1)
            kept = (elem * known).sum() / known.sum().clamp(min=1)
            pixel_loss = recon + self.known_pixel_weight * kept
            components[f"{name}_recon"] = recon.detach()
            components[f"{name}_known"] = kept.detach()
        else:
            pixel_loss = elem.mean()
        components[name] = pixel_loss.detach()
        total = self.l1_weight * pixel_loss

        if self.color_bias is not None:
            cb = self.color_bias(pred.float(), target.float())
            components["color_bias"] = cb.detach()
            total = total + self.color_bias_weight * cb

        components["total"] = total.detach()
        return total, components
