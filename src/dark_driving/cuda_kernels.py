"""CUDA-accelerated kernels for DarkDriving pipeline.

Provides fused operations for image preprocessing, augmentation,
and metric computation on GPU — avoiding CPU-GPU data transfers.

Uses torch.compile + custom CUDA kernels where available.
Falls back to PyTorch ops on CPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as f

# ---------------------------------------------------------------------------
# Fused image preprocessing (normalize + resize in one kernel)
# ---------------------------------------------------------------------------


@torch.compile(mode="reduce-overhead", fullgraph=True)
def fused_preprocess_cuda(
    images: torch.Tensor,
    target_h: int = 512,
    target_w: int = 512,
) -> torch.Tensor:
    """Fused GPU preprocessing: uint8->float32 + normalize + resize.

    Args:
        images: (B, C, H, W) uint8 tensor on CUDA.
        target_h: Target height.
        target_w: Target width.

    Returns:
        (B, C, target_h, target_w) float32 tensor in [0, 1].
    """
    x = images.float().div_(255.0)
    if x.shape[-2] != target_h or x.shape[-1] != target_w:
        x = f.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return x


# ---------------------------------------------------------------------------
# Fused augmentations (all on GPU, no CPU roundtrip)
# ---------------------------------------------------------------------------


@torch.compile(mode="reduce-overhead", fullgraph=True)
def fused_random_flip_cuda(
    night: torch.Tensor,
    day: torch.Tensor,
    flip_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused horizontal flip on GPU using pre-generated mask.

    Args:
        night: (B, C, H, W) night images.
        day: (B, C, H, W) day images.
        flip_mask: (B,) bool tensor — True means flip.

    Returns:
        Flipped (night, day) pair.
    """
    mask = flip_mask.view(-1, 1, 1, 1)
    night_flipped = torch.where(mask, night.flip(-1), night)
    day_flipped = torch.where(mask, day.flip(-1), day)
    return night_flipped, day_flipped


@torch.compile(mode="reduce-overhead", fullgraph=True)
def fused_random_crop_cuda(
    night: torch.Tensor,
    day: torch.Tensor,
    crop_h: int,
    crop_w: int,
    top: torch.Tensor,
    left: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused random crop on GPU using grid_sample for differentiable cropping.

    For fixed-size crops this is faster than CPU-based slicing.

    Args:
        night: (B, C, H, W) night images.
        day: (B, C, H, W) day images.
        crop_h: Crop height.
        crop_w: Crop width.
        top: (B,) top-left row indices.
        left: (B,) top-left col indices.
    """
    b, c, h, w = night.shape

    # Build sampling grid
    # Normalize coordinates to [-1, 1]
    grid_y = torch.linspace(0, crop_h - 1, crop_h, device=night.device)
    grid_x = torch.linspace(0, crop_w - 1, crop_w, device=night.device)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

    # Expand for batch
    grid_y = grid_y.unsqueeze(0).expand(b, -1, -1) + top.view(b, 1, 1)
    grid_x = grid_x.unsqueeze(0).expand(b, -1, -1) + left.view(b, 1, 1)

    # Normalize to [-1, 1]
    grid_y = 2.0 * grid_y / (h - 1) - 1.0
    grid_x = 2.0 * grid_x / (w - 1) - 1.0

    grid = torch.stack([grid_x, grid_y], dim=-1)

    night_crop = f.grid_sample(night, grid, mode="bilinear", align_corners=True)
    day_crop = f.grid_sample(day, grid, mode="bilinear", align_corners=True)

    return night_crop, day_crop


# ---------------------------------------------------------------------------
# Fused metric computation (PSNR + SSIM in one pass)
# ---------------------------------------------------------------------------


@torch.compile(mode="reduce-overhead", fullgraph=True)
def fused_psnr_ssim_cuda(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PSNR and SSIM in a single fused kernel on GPU.

    Returns:
        (psnr, ssim) as scalar tensors.
    """
    # PSNR
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
    psnr_mean = psnr.mean()

    # SSIM
    c1 = 0.01**2
    c2 = 0.03**2
    channels = pred.shape[1]

    # Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords = coords - window_size // 2
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = window.expand(channels, 1, -1, -1)

    pad = window_size // 2
    mu1 = f.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = f.conv2d(target, window, padding=pad, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = f.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = f.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = f.conv2d(pred * target, window, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    ssim_mean = ssim_map.mean()

    return psnr_mean, ssim_mean


# ---------------------------------------------------------------------------
# Fused loss computation (L1 + optional SSIM in one kernel)
# ---------------------------------------------------------------------------


@torch.compile(mode="reduce-overhead", fullgraph=True)
def fused_l1_ssim_loss_cuda(
    pred: torch.Tensor,
    target: torch.Tensor,
    ssim_weight: float = 0.0,
    window_size: int = 11,
) -> torch.Tensor:
    """Fused L1 + SSIM loss computation on GPU.

    Args:
        pred: (B, C, H, W) predicted images.
        target: (B, C, H, W) ground truth images.
        ssim_weight: Weight for SSIM loss (0 = L1 only).

    Returns:
        Scalar loss tensor.
    """
    l1_loss = f.l1_loss(pred, target)

    if ssim_weight <= 0.0:
        return l1_loss

    # SSIM computation
    c1 = 0.01**2
    c2 = 0.03**2
    channels = pred.shape[1]

    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords = coords - window_size // 2
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = window.expand(channels, 1, -1, -1)

    pad = window_size // 2
    mu1 = f.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = f.conv2d(target, window, padding=pad, groups=channels)

    sigma1_sq = f.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1 * mu1
    sigma2_sq = f.conv2d(target * target, window, padding=pad, groups=channels) - mu2 * mu2
    sigma12 = f.conv2d(pred * target, window, padding=pad, groups=channels) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1 * mu1 + mu2 * mu2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    ssim_loss = 1.0 - ssim_map.mean()

    return l1_loss + ssim_weight * ssim_loss


# ---------------------------------------------------------------------------
# CUDA dataset — loads images directly to GPU
# ---------------------------------------------------------------------------


class CUDAImageBatch:
    """Pre-loaded image batch on GPU for zero-copy training.

    Stores entire dataset in GPU memory (only viable for small datasets
    or subset caching). For DarkDriving's 9.5K pairs at 512x512:
    ~9538 * 2 * 3 * 512 * 512 * 4 bytes ≈ 56GB — too large for L4.

    Instead, use as a prefetch cache for the current epoch's mini-batches.
    """

    def __init__(self, device: torch.device, max_cached: int = 64):
        self.device = device
        self.max_cached = max_cached
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def put(self, idx: int, night: torch.Tensor, day: torch.Tensor) -> None:
        if len(self._cache) >= self.max_cached:
            # Evict oldest
            oldest = min(self._cache.keys())
            del self._cache[oldest]
        self._cache[idx] = (
            night.to(self.device, non_blocking=True),
            day.to(self.device, non_blocking=True),
        )

    def get(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._cache.get(idx)

    def clear(self) -> None:
        self._cache.clear()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Utility: check shared CUDA kernels availability
# ---------------------------------------------------------------------------


def check_shared_kernels() -> dict[str, bool]:
    """Check which shared CUDA kernels from shared_infra are available."""
    import importlib

    kernels = {
        "deformable_attention": "MultiScaleDeformableAttention",
        "fused_image_preprocess": "fused_preprocess",
        "vectorized_nms": "vectorized_nms",
    }
    results = {}
    for name, _ in kernels.items():
        try:
            importlib.import_module(name)
            results[name] = True
        except ImportError:
            results[name] = False
    return results
