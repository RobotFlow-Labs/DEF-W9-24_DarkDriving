#!/usr/bin/env python3
"""CUDA-accelerated training for DarkDriving.

Uses torch.compile, fused kernels, and GPU-native augmentations
for maximum throughput on L4 GPUs.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_cuda.py --config configs/paper.toml
    CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/train_cuda.py \
        --config configs/paper.toml --compile
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dark_driving.cuda_kernels import fused_l1_ssim_loss_cuda, fused_psnr_ssim_cuda
from dark_driving.dataset import build_dataloaders
from dark_driving.losses import build_loss
from dark_driving.model import count_parameters, get_model
from dark_driving.utils import (
    CheckpointManager,
    EarlyStopping,
    WarmupCosineScheduler,
    load_config,
    save_metrics_jsonl,
    set_seed,
)


def train_cuda(
    config_path: str,
    resume: str | None = None,
    max_steps: int | None = None,
    use_compile: bool = False,
) -> None:
    config = load_config(config_path)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})
    aug_cfg = config.get("augmentation", {})
    ckpt_cfg = config.get("checkpoint", {})
    es_cfg = config.get("early_stopping", {})
    log_cfg = config.get("logging", {})

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

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

    n_params = count_parameters(model)
    print(f"[CONFIG] {config_path}")
    print(f"[MODEL] {model_cfg.get('name', 'retinexformer')} — {n_params/1e6:.1f}M params")
    print(f"[DEVICE] {device} ({torch.cuda.get_device_name(0)})")

    # torch.compile for CUDA graph acceleration
    if use_compile and device.type == "cuda":
        print("[COMPILE] Compiling model with torch.compile (reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead")
        print("[COMPILE] Done — first forward pass will be slow (compiling)")

    # Data
    input_size = tuple(data_cfg.get("input_size", [512, 512]))
    batch_size = train_cfg.get("batch_size", 8)
    loaders = build_dataloaders(
        root=data_cfg.get("root", "/mnt/forge-data/datasets/darkdriving/"),
        input_size=input_size,
        batch_size=batch_size,
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

    print(f"[BATCH] batch_size={batch_size}")
    print(f"[DATA] train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)}")

    # Loss — use fused CUDA loss when possible
    use_fused_loss = loss_cfg.get("lpips_weight", 0.0) == 0.0 and \
                     loss_cfg.get("perceptual_weight", 0.0) == 0.0 and \
                     loss_cfg.get("charbonnier_weight", 0.0) == 0.0
    ssim_weight = loss_cfg.get("ssim_weight", 0.0)

    if use_fused_loss:
        print(f"[LOSS] Fused CUDA L1+SSIM (ssim_weight={ssim_weight})")
    else:
        print("[LOSS] Standard combined loss (has non-fusable components)")

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
    print(f"[TRAIN] {epochs} epochs, {steps_per_epoch} steps/epoch, "
          f"lr={lr}, warmup={warmup_steps} steps")

    # AMP
    precision = train_cfg.get("precision", "fp16")
    use_amp = precision in ("fp16", "bf16") and device.type == "cuda"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and precision == "fp16")
    print(f"[AMP] precision={precision}, autocast={'ON' if use_amp else 'OFF'}")

    # Checkpoint manager
    ckpt_dir = ckpt_cfg.get("output_dir", "/mnt/artifacts-datai/checkpoints/project_darkdriving")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_manager = CheckpointManager(
        save_dir=ckpt_dir,
        keep_top_k=ckpt_cfg.get("keep_top_k", 2),
        metric=ckpt_cfg.get("metric", "val_psnr"),
        mode=ckpt_cfg.get("mode", "max"),
    )
    print(f"[CKPT] save to {ckpt_dir}, keep top {ckpt_cfg.get('keep_top_k', 2)}")

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
            pass

    # Resume
    start_epoch = 0
    global_step = 0
    if resume:
        print(f"[RESUME] Loading checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        # Handle compiled model state dict
        state_dict = ckpt["model"]
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # Strip _orig_mod prefix from compiled model
            clean = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean)
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        print(f"[RESUME] Continuing from epoch {start_epoch}, step {global_step}")

    # Metrics log
    metrics_path = Path(log_cfg.get("log_dir", ckpt_dir)) / "metrics.jsonl"
    os.makedirs(metrics_path.parent, exist_ok=True)

    # CUDA optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ===== TRAINING LOOP =====
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    best_metric = float("-inf")

    print(f"\n{'='*60}")
    print(f"CUDA ACCELERATED TRAINING — {model_cfg.get('name', 'retinexformer')}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for batch in tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            night = batch["night"].to(device, non_blocking=True)
            day = batch["day"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                enhanced = model(night)
                if use_fused_loss:
                    loss = fused_l1_ssim_loss_cuda(enhanced, day, ssim_weight)
                    loss_dict = {"l1": loss.item(), "total": loss.item()}
                else:
                    loss, loss_dict = loss_fn(enhanced, day)

            if torch.isnan(loss):
                print("[FATAL] Loss is NaN — stopping training")
                sys.exit(1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_sum += loss.item()
            n_batches += 1
            global_step += 1

            if tb_writer and global_step % 10 == 0:
                tb_writer.add_scalar("train/loss", loss.item(), global_step)
                tb_writer.add_scalar("train/lr", scheduler.get_lr(), global_step)

            if max_steps and global_step >= max_steps:
                break

        train_loss = train_loss_sum / max(n_batches, 1)

        # Validate with fused CUDA metrics
        model.eval()
        val_loss_sum = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(loaders["val"], desc="Val", leave=False):
                night = batch["night"].to(device, non_blocking=True)
                day = batch["day"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                    enhanced = model(night)
                    if use_fused_loss:
                        loss = fused_l1_ssim_loss_cuda(enhanced, day, ssim_weight)
                    else:
                        loss, _ = loss_fn(enhanced, day)

                enhanced_clamped = enhanced.clamp(0.0, 1.0)
                psnr, ssim = fused_psnr_ssim_cuda(enhanced_clamped, day)

                val_loss_sum += loss.item()
                val_psnr_sum += psnr.item()
                val_ssim_sum += ssim.item()
                val_batches += 1

        vn = max(val_batches, 1)
        val_loss = val_loss_sum / vn
        val_psnr = val_psnr_sum / vn
        val_ssim = val_ssim_sum / vn
        elapsed = time.time() - t0
        throughput = n_batches * batch_size / elapsed

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_psnr={val_psnr:.2f} "
            f"val_ssim={val_ssim:.4f} "
            f"lr={scheduler.get_lr():.2e} "
            f"{throughput:.0f} img/s "
            f"time={elapsed:.1f}s"
        )

        if tb_writer:
            tb_writer.add_scalar("val/loss", val_loss, epoch + 1)
            tb_writer.add_scalar("val/psnr", val_psnr, epoch + 1)
            tb_writer.add_scalar("val/ssim", val_ssim, epoch + 1)
            tb_writer.add_scalar("train/throughput", throughput, epoch + 1)

        save_metrics_jsonl(metrics_path, {
            "epoch": epoch + 1, "step": global_step,
            "train_loss": train_loss, "val_loss": val_loss,
            "val_psnr": val_psnr, "val_ssim": val_ssim,
            "lr": scheduler.get_lr(), "throughput": throughput, "time_s": elapsed,
        })

        # Checkpoint — get raw model state dict (unwrap compiled if needed)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        state = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "step": global_step,
            "metrics": {"val_loss": val_loss, "val_psnr": val_psnr, "val_ssim": val_ssim},
            "config": config,
        }
        ckpt_manager.save(state, val_psnr, global_step)

        if val_psnr > best_metric:
            best_metric = val_psnr

        # Early stopping
        if early_stopper and early_stopper.step(val_psnr):
            print(f"[EARLY STOP] No improvement for {early_stopper.patience} epochs.")
            break

        if max_steps and global_step >= max_steps:
            print(f"[MAX STEPS] Reached {max_steps} steps.")
            break

    if tb_writer:
        tb_writer.close()

    print(f"\n[DONE] Training complete. Best val_psnr: {best_metric:.2f}")
    print(f"[DONE] Checkpoints: {ckpt_dir}")

    # Cleanup
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA-accelerated DarkDriving training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    args = parser.parse_args()

    train_cuda(args.config, args.resume, args.max_steps, args.compile)


if __name__ == "__main__":
    main()
