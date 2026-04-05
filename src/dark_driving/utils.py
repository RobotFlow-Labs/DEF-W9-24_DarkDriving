"""Utility functions for DarkDriving module.

Config loading, metric computation, visualization, and checkpoint management.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    import tomli
except ImportError:
    try:
        import tomllib as tomli  # type: ignore[no-redef]
    except ImportError:
        tomli = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML config file and return as nested dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    if tomli is not None:
        with open(path, "rb") as f:
            return tomli.load(f)

    # Fallback: try tomllib (Python 3.11+)
    import tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)


def flat_config(cfg: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config dict to dot-separated keys."""
    flat: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flat_config(v, key))
        else:
            flat[key] = v
    return flat


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: (B, C, H, W) predicted image in [0, max_val].
        target: (B, C, H, W) ground truth image in [0, max_val].
        max_val: Maximum pixel value.

    Returns:
        PSNR in dB (averaged over batch).
    """
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr = 10.0 * torch.log10(max_val**2 / (mse + 1e-10))
    return psnr.mean().item()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> float:
    """Compute Structural Similarity Index.

    Uses Gaussian windowed approach matching the paper's evaluation.
    """
    c1 = 0.01**2
    c2 = 0.03**2
    channels = pred.shape[1]

    # Create Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords = coords - window_size // 2
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)

    pad = window_size // 2

    mu1 = torch.nn.functional.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = torch.nn.functional.conv2d(target, window, padding=pad, groups=channels)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        torch.nn.functional.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
    )
    sigma2_sq = (
        torch.nn.functional.conv2d(target * target, window, padding=pad, groups=channels)
        - mu2_sq
    )
    sigma12 = (
        torch.nn.functional.conv2d(pred * target, window, padding=pad, groups=channels)
        - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Manages model checkpoints: saves top-K by metric, auto-deletes old ones.

    Follows ANIMA training standards:
    - Keep only top K checkpoints ranked by validation metric
    - Always maintain a separate best.pth
    - Save full state: model, optimizer, scheduler, epoch, step, metrics, config
    """

    def __init__(
        self,
        save_dir: str | Path,
        keep_top_k: int = 2,
        metric: str = "val_psnr",
        mode: str = "max",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(
        self,
        state: dict[str, Any],
        metric_value: float,
        step: int,
    ) -> Path:
        """Save checkpoint and prune old ones.

        Args:
            state: Dict with model, optimizer, scheduler state_dicts, etc.
            metric_value: Value of the tracked metric.
            step: Current training step.

        Returns:
            Path to saved checkpoint.
        """
        path = self.save_dir / f"checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))

        # Sort: best first
        reverse = self.mode == "max"
        self.history.sort(key=lambda x: x[0], reverse=reverse)

        # Prune
        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            if old_path.exists():
                old_path.unlink()

        # Save best separately
        best_val, best_path = self.history[0]
        best_dst = self.save_dir / "best.pth"
        if best_path != best_dst:
            shutil.copy2(best_path, best_dst)

        return path

    @property
    def best_metric(self) -> float | None:
        if not self.history:
            return None
        return self.history[0][0]


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Early stopping tracker.

    Stops training when the monitored metric hasn't improved for `patience` epochs.
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.001,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        """Check if training should stop.

        Returns:
            True if training should stop (patience exhausted).
        """
        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            return True
        return False


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------


class WarmupCosineScheduler:
    """Cosine annealing with linear warmup.

    Paper uses: 5% warmup + cosine decay to min_lr.
    Resume-aware: saves/restores current step.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / max(1, self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=False):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict[str, Any]:
        return {"current_step": self.current_step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.current_step = state["current_step"]


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log_training_start(config: dict, model: nn.Module, device: str) -> None:
    """Print training configuration summary at start."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[CONFIG] {config.get('_config_path', 'unknown')}")
    model_name = config.get("model", {}).get("name", "unknown")
    print(f"[MODEL] {model_name} -- {n_params / 1e6:.1f}M params")
    print(f"[DEVICE] {device}")
    print(f"[BATCH] batch_size={config.get('training', {}).get('batch_size', '?')}")
    print(f"[TRAIN] epochs={config.get('training', {}).get('epochs', '?')}, "
          f"lr={config.get('training', {}).get('learning_rate', '?')}")
    print(f"[CKPT] save to {config.get('checkpoint', {}).get('output_dir', '?')}")


def save_metrics_jsonl(path: str | Path, metrics: dict[str, Any]) -> None:
    """Append metrics dict as a JSON line to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(metrics) + "\n")
