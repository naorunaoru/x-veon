"""
Advanced loss functions for X-Trans demosaicing.

- Gradient loss (Sobel): penalizes edge smearing
- Chromatic artifact penalty (Lab): penalizes high-frequency chroma (false color)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelGradientLoss(nn.Module):
    """
    Compare spatial gradients between prediction and ground truth.
    Uses Sobel filters to extract horizontal and vertical edges.
    Penalizes edge smearing and ringing.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels (3x3)
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1],
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Register as buffers (not parameters, but move to device with model)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude per channel. x: (B, C, H, W)"""
        B, C, H, W = x.shape
        # Apply Sobel per channel
        x_flat = x.reshape(B * C, 1, H, W)
        gx = F.conv2d(x_flat, self.sobel_x, padding=1)
        gy = F.conv2d(x_flat, self.sobel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return mag.reshape(B, C, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_grad = self._gradient_magnitude(pred)
        target_grad = self._gradient_magnitude(target)
        return F.l1_loss(pred_grad, target_grad)


class ChromaticArtifactLoss(nn.Module):
    """
    Penalize high-frequency content in chrominance channels.
    
    Converts to a luminance/chrominance space, applies high-pass filtering
    to the chroma channels, and penalizes the result.
    
    Key insight: real image detail lives in luminance. High-frequency
    chrominance is almost always a demosaicing artifact (zipper, false color,
    green/magenta fringing).
    
    Uses YCbCr instead of Lab for differentiability (linear transform).
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        # Gaussian low-pass kernel for extracting low-freq component
        # High-freq = original - low-pass
        sigma = kernel_size / 4.0
        ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
        self.register_buffer('lowpass', kernel)

    def _rgb_to_ycbcr(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB [0,1] to YCbCr. Returns (B, 3, H, W)."""
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        y  =  0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.169 * r - 0.331 * g + 0.500 * b
        cr =  0.500 * r - 0.419 * g - 0.081 * b
        return torch.cat([y, cb, cr], dim=1)

    def _highpass_chroma(self, ycbcr: torch.Tensor) -> torch.Tensor:
        """Extract high-frequency content from Cb and Cr channels."""
        chroma = ycbcr[:, 1:3]  # (B, 2, H, W)
        B, C, H, W = chroma.shape
        chroma_flat = chroma.reshape(B * C, 1, H, W)
        pad = self.kernel_size // 2
        lowfreq = F.conv2d(chroma_flat, self.lowpass, padding=pad)
        highfreq = chroma_flat - lowfreq
        return highfreq.reshape(B, C, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_ycbcr = self._rgb_to_ycbcr(pred)
        target_ycbcr = self._rgb_to_ycbcr(target)
        
        pred_hf = self._highpass_chroma(pred_ycbcr)
        target_hf = self._highpass_chroma(target_ycbcr)
        
        # Penalize excess high-frequency chroma in prediction vs target
        return F.l1_loss(pred_hf, target_hf)


class CombinedLoss(nn.Module):
    """
    Combined loss: L1 + gradient + chromatic artifact penalty.
    
    Args:
        gradient_weight: weight for Sobel gradient loss (default 0.1)
        chroma_weight: weight for chromatic artifact loss (default 0.05)
    """

    def __init__(self, gradient_weight: float = 0.1, chroma_weight: float = 0.05):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.gradient = SobelGradientLoss()
        self.chroma = ChromaticArtifactLoss()
        self.gradient_weight = gradient_weight
        self.chroma_weight = chroma_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        l1_loss = self.l1(pred, target)
        grad_loss = self.gradient(pred, target)
        chroma_loss = self.chroma(pred, target)
        
        total = l1_loss + self.gradient_weight * grad_loss + self.chroma_weight * chroma_loss
        
        components = {
            'l1': l1_loss.item(),
            'gradient': grad_loss.item(),
            'chroma': chroma_loss.item(),
            'total': total.item(),
        }
        return total, components


class SSIM(nn.Module):
    """Structural Similarity Index for perceptual quality."""
    
    def __init__(self, window_size=11, sigma=1.5, channels=3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        
        # Create Gaussian kernel
        x = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = gauss.outer(gauss)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)
        self.register_buffer('kernel', kernel)
        
    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2
        
        mu1 = F.conv2d(pred, self.kernel, padding=pad, groups=self.channels)
        mu2 = F.conv2d(target, self.kernel, padding=pad, groups=self.channels)
        
        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, self.kernel, padding=pad, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.kernel, padding=pad, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.kernel, padding=pad, groups=self.channels) - mu1_mu2
        
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim.mean()


class SSIMGradientLoss(nn.Module):
    """Combined L1 + SSIM + Gradient loss for texture preservation."""
    
    def __init__(self, l1_weight=0.4, ssim_weight=0.5, gradient_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
        self.ssim = SSIM()
        self.gradient = SobelGradientLoss()
    
    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ssim = 1 - self.ssim(pred, target)  # 1 - SSIM so lower is better
        grad = self.gradient(pred, target)
        
        total = self.l1_weight * l1 + self.ssim_weight * ssim + self.gradient_weight * grad
        components = {'l1': l1.item(), 'ssim': (1-ssim).item(), 'grad': grad.item()}
        return total, components
