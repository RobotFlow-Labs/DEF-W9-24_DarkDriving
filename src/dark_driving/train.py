"""Training pipeline for DarkDriving enhancement models.

Config-driven training loop with:
- Mixed precision (AMP)
- Cosine annealing + warmup LR schedule
- Checkpoint management (keep best K)
- Early stopping
- TensorBoard logging
- Resume from checkpoint
- NaN detection
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dark_driving.dataset import build_dataloaders
from dark_driving.losses import build_loss
from dark_driving.model import get_model
from dark_driving.utils import (
    CheckpointManager,
    EarlyStopping,
    WarmupCosineScheduler,
    compute_psnr,
    compute_ssim,
    load_config,
    log_training_start,
    save_metrics_jsonl,
    set_seed,
)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    global_step: int = 0,
    ckpt_manager: CheckpointManager | None = None,
    save_every: int = 500,
    tb_writer: Any = None,
) -> tuple[float, int]:
    """Run one training epoch.

    Returns:
        (average_loss, updated_global_step)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        night = batch["night"].to(device, non_blocking=True)
        day = batch["day"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            enhanced = model(night)
            loss, loss_dict = loss_fn(enhanced, day)

        # NaN detection
        if torch.isnan(loss):
            print("[FATAL] Loss is NaN -- stopping training")
            print("[FIX] Reduce lr by 10x, check data for corrupt samples")
            sys.exit(1)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        global_step += 1

        # TensorBoard logging
        if tb_writer is not None and global_step % 10 == 0:
            tb_writer.add_scalar("train/loss", loss.item(), global_step)
            tb_writer.add_scalar("train/lr", scheduler.get_lr(), global_step)
            for k, v in loss_dict.items():
                if k != "total":
                    tb_writer.add_scalar(f"train/loss_{k}", v, global_step)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> dict[str, float]:
    """Run validation and compute metrics.

    Returns:
        Dict with val_loss, val_psnr, val_ssim.
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        night = batch["night"].to(device, non_blocking=True)
        day = batch["day"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            enhanced = model(night)
            loss, _ = loss_fn(enhanced, day)

        # Clamp to [0, 1] for metric computation
        enhanced_clamped = enhanced.clamp(0.0, 1.0)

        total_loss += loss.item()
        total_psnr += compute_psnr(enhanced_clamped, day)
        total_ssim += compute_ssim(enhanced_clamped, day)
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "val_loss": total_loss / n,
        "val_psnr": total_psnr / n,
        "val_ssim": total_ssim / n,
    }


def train(config_path: str, resume: str | None = None, max_steps: int | None = None) -> None:
    """Main training function.

    Args:
        config_path: Path to TOML config file.
        resume: Optional path to checkpoint for resuming.
        max_steps: Optional max steps (for smoke tests).
    """
    config = load_config(config_path)
    config["_config_path"] = config_path

    # Extract config sections
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})
    aug_cfg = config.get("augmentation", {})
    ckpt_cfg = config.get("checkpoint", {})
    es_cfg = config.get("early_stopping", {})
    log_cfg = config.get("logging", {})

    # Seed
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = get_model(
        name=model_cfg.get("name", "retinexformer"),
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 3),
        embed_dim=model_cfg.get("embed_dim", 32),
        num_blocks=model_cfg.get("num_blocks", 4),
        num_heads=model_cfg.get("num_heads", 4),
    ).to(device)

    log_training_start(config, model, str(device))

    # Data
    input_size = tuple(data_cfg.get("input_size", [512, 512]))
    loaders = build_dataloaders(
        root=data_cfg.get("root", "/mnt/forge-data/datasets/darkdriving/"),
        input_size=input_size,
        batch_size=train_cfg.get("batch_size", 8),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        seed=seed,
        augment_train=True,
        crop_size=tuple(aug_cfg.get("crop_size", list(input_size))),
        horizontal_flip=aug_cfg.get("horizontal_flip", True),
        flip_prob=aug_cfg.get("flip_prob", 0.5),
        random_rotation=aug_cfg.get("random_rotation", True),
        rotation_degrees=aug_cfg.get("rotation_degrees", 90),
    )

    # Loss
    loss_fn = build_loss(loss_cfg).to(device)

    # Optimizer
    lr = train_cfg.get("learning_rate", 1e-3)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(train_cfg.get("beta1", 0.9), train_cfg.get("beta2", 0.999)),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    # Scheduler
    epochs = train_cfg.get("epochs", 200)
    steps_per_epoch = max(len(loaders["train"]), 1)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.05))
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps, total_steps, min_lr=train_cfg.get("min_lr", 1e-6)
    )

    # AMP
    use_amp = train_cfg.get("precision", "fp16") in ("fp16", "bf16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Checkpoint manager
    ckpt_dir = ckpt_cfg.get("output_dir", "/mnt/artifacts-datai/checkpoints/project_darkdriving")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_manager = CheckpointManager(
        save_dir=ckpt_dir,
        keep_top_k=ckpt_cfg.get("keep_top_k", 2),
        metric=ckpt_cfg.get("metric", "val_psnr"),
        mode=ckpt_cfg.get("mode", "max"),
    )

    # Early stopping
    early_stopper = None
    if es_cfg.get("enabled", True):
        early_stopper = EarlyStopping(
            patience=es_cfg.get("patience", 20),
            min_delta=es_cfg.get("min_delta", 0.001),
            mode=ckpt_cfg.get("mode", "max"),
        )

    # TensorBoard
    tb_writer = None
    tb_dir = log_cfg.get("tensorboard_dir")
    if tb_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(tb_dir)
        except ImportError:
            print("[WARN] TensorBoard not available")

    # Resume
    start_epoch = 0
    global_step = 0
    if resume:
        print(f"[RESUME] Loading checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        print(f"[RESUME] Continuing from epoch {start_epoch}, step {global_step}")

    # Metrics log path
    metrics_path = Path(log_cfg.get("log_dir", ckpt_dir)) / "metrics.jsonl"
    os.makedirs(metrics_path.parent, exist_ok=True)

    # Training loop
    print(f"\n[TRAIN] Starting training: {epochs} epochs, {steps_per_epoch} steps/epoch")
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss, global_step = train_one_epoch(
            model=model,
            loader=loaders["train"],
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
            use_amp=use_amp,
            global_step=global_step,
            ckpt_manager=ckpt_manager,
            save_every=ckpt_cfg.get("save_every_n_steps", 500),
            tb_writer=tb_writer,
        )

        # Validate
        val_metrics = validate(model, loaders["val"], loss_fn, device, use_amp)
        elapsed = time.time() - t0

        # Log
        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"val_psnr={val_metrics['val_psnr']:.2f} "
            f"val_ssim={val_metrics['val_ssim']:.4f} "
            f"lr={scheduler.get_lr():.2e} "
            f"time={elapsed:.1f}s"
        )

        # TensorBoard epoch metrics
        if tb_writer:
            for k, v in val_metrics.items():
                tb_writer.add_scalar(f"val/{k}", v, epoch + 1)

        # Save metrics
        save_metrics_jsonl(metrics_path, {
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": train_loss,
            **val_metrics,
            "lr": scheduler.get_lr(),
            "time_s": elapsed,
        })

        # Checkpoint
        metric_val = val_metrics.get(ckpt_cfg.get("metric", "val_psnr"), 0.0)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "step": global_step,
            "metrics": val_metrics,
            "config": config,
        }
        ckpt_manager.save(state, metric_val, global_step)

        # Early stopping
        if early_stopper is not None and early_stopper.step(metric_val):
            print(f"[EARLY STOP] No improvement for {early_stopper.patience} epochs. Stopping.")
            break

        # Max steps (for smoke tests)
        if max_steps is not None and global_step >= max_steps:
            print(f"[MAX STEPS] Reached {max_steps} steps. Stopping.")
            break

    if tb_writer:
        tb_writer.close()

    print(f"\n[DONE] Training complete. Best {ckpt_cfg.get('metric', 'val_psnr')}: "
          f"{ckpt_manager.best_metric}")
    print(f"[DONE] Checkpoints saved to: {ckpt_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DarkDriving enhancement training")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resume")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps (smoke test)")
    args = parser.parse_args()

    train(args.config, resume=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
