#!/usr/bin/env python3
"""GPU batch size auto-detection for DarkDriving.

Finds the optimal batch size targeting 65% VRAM utilization on L4 (23GB).
Runs binary search: allocates model + data, measures VRAM, adjusts.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/find_batch_size.py --target 0.65
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/find_batch_size.py \
        --model retinexformer --target 0.70
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dark_driving.model import get_model


def find_optimal_batch(
    model_name: str = "retinexformer",
    embed_dim: int = 32,
    num_blocks: int = 4,
    num_heads: int = 4,
    input_size: tuple[int, int] = (512, 512),
    target_util: float = 0.65,
    max_batch: int = 128,
    device_id: int = 0,
) -> int:
    device = torch.device(f"cuda:{device_id}")
    total_mem = torch.cuda.get_device_properties(device).total_mem
    target_bytes = int(total_mem * target_util)

    print(f"[BATCH FINDER] GPU: {torch.cuda.get_device_name(device)}")
    print(f"[BATCH FINDER] Total VRAM: {total_mem / 1e9:.1f}GB")
    print(f"[BATCH FINDER] Target: {target_util*100:.0f}% = {target_bytes / 1e9:.1f}GB")

    model = get_model(
        name=model_name,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")

    lo, hi = 1, max_batch
    best_bs = 1

    while lo <= hi:
        bs = (lo + hi) // 2
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)

        try:
            x = torch.randn(bs, 3, *input_size, device=device)
            target = torch.randn_like(x)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = torch.nn.functional.l1_loss(out, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            peak = torch.cuda.max_memory_allocated(device)
            util = peak / total_mem

            print(f"  bs={bs:4d}  peak={peak/1e9:.2f}GB  util={util*100:.1f}%", end="")

            if peak <= target_bytes:
                best_bs = bs
                lo = bs + 1
                print("  ✓ fits")
            else:
                hi = bs - 1
                print("  ✗ over target")

            del x, target, out, loss

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  bs={bs:4d}  OOM")
                hi = bs - 1
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise

    # Final check
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    x = torch.randn(best_bs, 3, *input_size, device=device)
    target = torch.randn_like(x)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        out = model(x)
        loss = torch.nn.functional.l1_loss(out, target)
    scaler.scale(loss).backward()
    peak = torch.cuda.max_memory_allocated(device)
    util = peak / total_mem

    del model, optimizer, scaler, x, target, out, loss
    torch.cuda.empty_cache()
    gc.collect()

    print(
        f"\n[BATCH FINDER] Optimal batch_size={best_bs}"
        f" ({util*100:.1f}% of {total_mem/1e9:.1f}GB)"
    )
    return best_bs


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU batch size finder for DarkDriving")
    parser.add_argument("--model", default="retinexformer", help="Model name")
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--input-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--target", type=float, default=0.65)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    bs = find_optimal_batch(
        model_name=args.model,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        input_size=tuple(args.input_size),
        target_util=args.target,
        max_batch=args.max_batch,
        device_id=args.device,
    )
    print(f"\nResult: batch_size={bs}")


if __name__ == "__main__":
    main()
