"""Loss functions for DarkDriving enhancement training.

Primary loss: L1 (MAE) between enhanced and ground-truth day image.
Optional: SSIM, LPIPS, Charbonnier, Perceptual (VGG).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f


class L1Loss(nn.Module):
    """Standard L1 (Mean Absolute Error) loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return f.l1_loss(pred, target)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1 variant).

    L = mean(sqrt((pred - target)^2 + eps^2))
    Used by Retinexformer as primary loss.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss: L = 1 - SSIM(pred, target).

    Uses a Gaussian window for local statistics computation.
    """

    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer("window", self._create_window(window_size, channels))

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        return g

    def _create_window(self, size: int, channels: int) -> torch.Tensor:
        _1d = self._gaussian_kernel(size)
        _2d = _1d.unsqueeze(1) * _1d.unsqueeze(0)
        window = _2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size)
        return window.contiguous()

    def _ssim(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        c1 = 0.01**2
        c2 = 0.03**2
        pad = self.window_size // 2

        window = self.window  # type: ignore[attr-defined]
        if window.device != pred.device:
            window = window.to(pred.device)

        mu1 = f.conv2d(pred, window, padding=pad, groups=self.channels)
        mu2 = f.conv2d(target, window, padding=pad, groups=self.channels)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = f.conv2d(pred * pred, window, padding=pad, groups=self.channels) - mu1_sq
        sigma2_sq = (
            f.conv2d(target * target, window, padding=pad, groups=self.channels) - mu2_sq
        )
        sigma12 = (
            f.conv2d(pred * target, window, padding=pad, groups=self.channels) - mu1_mu2
        )

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._ssim(pred, target)


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS) loss.

    Uses VGG features for perceptual similarity measurement.
    Wraps the lpips package if available, otherwise falls back to VGG L2.
    """

    def __init__(self, net: str = "vgg"):
        super().__init__()
        self._lpips = None
        self._net = net

    def _init_lpips(self, device: torch.device) -> None:
        """Lazy initialization of LPIPS model."""
        try:
            import lpips

            self._lpips = lpips.LPIPS(net=self._net).to(device)
            self._lpips.eval()
            for p in self._lpips.parameters():
                p.requires_grad = False
        except ImportError:
            self._lpips = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._lpips is None:
            self._init_lpips(pred.device)

        if self._lpips is not None:
            # LPIPS expects [-1, 1] range
            pred_scaled = pred * 2.0 - 1.0
            target_scaled = target * 2.0 - 1.0
            return self._lpips(pred_scaled, target_scaled).mean()

        # Fallback: simple L2 in pixel space
        return f.mse_loss(pred, target)


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss (feature matching).

    Extracts features from VGG-19 layers and computes L1 distance.
    """

    def __init__(self, layers: list[int] | None = None):
        super().__init__()
        self.layers = layers or [3, 8, 17, 26]  # relu1_2, relu2_2, relu3_4, relu4_4
        self._vgg = None

    def _init_vgg(self, device: torch.device) -> None:
        """Lazy VGG initialization."""
        try:
            from torchvision.models import VGG19_Weights, vgg19

            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device)
            vgg.eval()
            for p in vgg.parameters():
                p.requires_grad = False
            self._vgg = vgg
        except Exception:
            self._vgg = None

    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        if self._vgg is None:
            return features
        for i, layer in enumerate(self._vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._vgg is None:
            self._init_vgg(pred.device)

        if self._vgg is None:
            return f.l1_loss(pred, target)

        pred_feats = self._extract_features(pred)
        target_feats = self._extract_features(target)

        loss = torch.tensor(0.0, device=pred.device)
        for pf, tf in zip(pred_feats, target_feats, strict=False):
            loss = loss + f.l1_loss(pf, tf)

        return loss / max(len(pred_feats), 1)


class CombinedLoss(nn.Module):
    """Weighted combination of multiple losses.

    Configurable via TOML config weights:
        l1_weight, ssim_weight, lpips_weight, charbonnier_weight, perceptual_weight
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.0,
        lpips_weight: float = 0.0,
        charbonnier_weight: float = 0.0,
        charbonnier_eps: float = 1e-3,
        perceptual_weight: float = 0.0,
    ):
        super().__init__()
        self.weights = {
            "l1": l1_weight,
            "ssim": ssim_weight,
            "lpips": lpips_weight,
            "charbonnier": charbonnier_weight,
            "perceptual": perceptual_weight,
        }

        self.losses: dict[str, nn.Module] = {}
        if l1_weight > 0:
            self.losses["l1"] = L1Loss()
        if ssim_weight > 0:
            self.losses["ssim"] = SSIMLoss()
        if lpips_weight > 0:
            self.losses["lpips"] = LPIPSLoss()
        if charbonnier_weight > 0:
            self.losses["charbonnier"] = CharbonnierLoss(eps=charbonnier_eps)
        if perceptual_weight > 0:
            self.losses["perceptual"] = PerceptualLoss()

        # Register as ModuleDict for proper device handling
        self._modules_dict = nn.ModuleDict(self.losses)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Returns:
            (total_loss, loss_dict) where loss_dict has per-component values.
        """
        total = torch.tensor(0.0, device=pred.device, requires_grad=True)
        loss_dict: dict[str, float] = {}

        for name, loss_fn in self.losses.items():
            w = self.weights[name]
            val = loss_fn(pred, target)
            total = total + w * val
            loss_dict[name] = val.item()

        loss_dict["total"] = total.item()
        return total, loss_dict


def build_loss(config: dict) -> CombinedLoss:
    """Build loss from config dict (parsed from TOML [loss] section).

    Args:
        config: Dict with keys like l1_weight, ssim_weight, etc.

    Returns:
        CombinedLoss instance.
    """
    return CombinedLoss(
        l1_weight=config.get("l1_weight", 1.0),
        ssim_weight=config.get("ssim_weight", 0.0),
        lpips_weight=config.get("lpips_weight", 0.0),
        charbonnier_weight=config.get("charbonnier_weight", 0.0),
        charbonnier_eps=config.get("charbonnier_eps", 1e-3),
        perceptual_weight=config.get("perceptual_weight", 0.0),
    )
