# PRD-04: Training Pipeline

## Objective
Implement the full training loop for enhancement model benchmarking.

## Deliverables
1. `src/dark_driving/train.py` -- Training entry point:
   - Config-driven (TOML)
   - Adam optimizer, cosine annealing LR with warmup
   - Mixed precision (AMP) on CUDA
   - Gradient clipping (max_norm=1.0)
   - Checkpoint manager (keep best 2 by val PSNR)
   - Early stopping (patience=20 epochs)
   - TensorBoard logging
   - Resume from checkpoint support
   - NaN loss detection
2. `scripts/train.py` -- CLI wrapper
3. `scripts/find_batch_size.py` -- GPU batch size auto-detection

## Training Flow
```
1. Load config (TOML)
2. Build dataset (train/val split from train set: 90/10)
3. Build model (from registry)
4. Build optimizer + scheduler
5. For each epoch:
   a. Train step: forward -> loss -> backward -> optimizer step
   b. Val step: compute PSNR/SSIM on val set
   c. Checkpoint if improved
   d. Early stop if plateaued
6. Save final best model
```

## Hyperparameters (from paper)
- Input: 512x512
- Batch size: 8 (or auto-detected)
- LR: 1e-3
- Optimizer: Adam (beta1=0.9, beta2=0.999)
- Scheduler: cosine annealing
- Warmup: 5% of total steps
- Epochs: 200
- Precision: fp16 (AMP)
- Gradient clipping: 1.0
- Seed: 42

## Acceptance Criteria
- Smoke test: 2-step training on random data completes without error
- Checkpoint save/load cycle works
- Resume from checkpoint produces correct epoch/step
- TensorBoard events are written
