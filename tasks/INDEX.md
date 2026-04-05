# Tasks Index -- DarkDriving

## PRD-01: Foundation
- [ ] T01.1: Create pyproject.toml with hatchling backend + all deps
- [ ] T01.2: Create anima_module.yaml manifest
- [ ] T01.3: Create configs/paper.toml with paper hyperparameters
- [ ] T01.4: Create configs/debug.toml for smoke testing
- [ ] T01.5: Create src/dark_driving/__init__.py
- [ ] T01.6: Create .venv and verify uv sync
- [ ] T01.7: Verify ruff check passes

## PRD-02: Core Model
- [ ] T02.1: Implement Retinexformer architecture (illumination estimator + IGT)
- [ ] T02.2: Implement SNR-Aware model wrapper
- [ ] T02.3: Implement LLFormer model wrapper
- [ ] T02.4: Implement generic UNet baseline
- [ ] T02.5: Create model registry (get_model_by_name)
- [ ] T02.6: Implement DarkDriving dataset loader (day-night pairs)
- [ ] T02.7: Implement augmentations (crop, rotate, flip)
- [ ] T02.8: Implement bbox annotation loader
- [ ] T02.9: Write tests/test_model.py
- [ ] T02.10: Write tests/test_dataset.py

## PRD-03: Loss Functions
- [ ] T03.1: Implement L1Loss wrapper
- [ ] T03.2: Implement differentiable SSIMLoss
- [ ] T03.3: Implement LPIPSLoss (VGG-based)
- [ ] T03.4: Implement CharbonnierLoss
- [ ] T03.5: Implement PerceptualLoss (VGG feature matching)
- [ ] T03.6: Implement CombinedLoss with configurable weights
- [ ] T03.7: Unit tests for all losses (gradient flow)

## PRD-04: Training Pipeline
- [ ] T04.1: Implement config loader (TOML -> Pydantic)
- [ ] T04.2: Implement training loop with AMP
- [ ] T04.3: Implement checkpoint manager (keep best 2)
- [ ] T04.4: Implement cosine annealing + warmup scheduler
- [ ] T04.5: Implement early stopping
- [ ] T04.6: Implement NaN detection + graceful stop
- [ ] T04.7: Implement TensorBoard logging
- [ ] T04.8: Implement resume from checkpoint
- [ ] T04.9: Create scripts/train.py CLI
- [ ] T04.10: Create scripts/find_batch_size.py
- [ ] T04.11: Smoke test: 2-step train + checkpoint round-trip

## PRD-05: Evaluation
- [ ] T05.1: Implement PSNR metric
- [ ] T05.2: Implement SSIM metric
- [ ] T05.3: Implement LPIPS metric
- [ ] T05.4: Implement no-reference metrics (MUSIQ, NIQE, HyperIQA, CNNIQA)
- [ ] T05.5: Implement 2D detection evaluation (AP50 via YOLOv11)
- [ ] T05.6: Implement per-condition / per-road-type breakdown
- [ ] T05.7: Implement report generation (JSON + markdown)
- [ ] T05.8: Create scripts/evaluate.py CLI

## PRD-06: Export Pipeline
- [ ] T06.1: Implement PyTorch -> SafeTensors export
- [ ] T06.2: Implement PyTorch -> ONNX export (opset 17)
- [ ] T06.3: Implement ONNX -> TensorRT FP16/FP32 (shared toolkit)
- [ ] T06.4: Implement output validation (diff < 1e-3)
- [ ] T06.5: Implement latency benchmarking

## PRD-07: Integration
- [ ] T07.1: Create Dockerfile.serve (3-layer)
- [ ] T07.2: Create docker-compose.serve.yml (profiles)
- [ ] T07.3: Create .env.serve
- [ ] T07.4: Implement src/dark_driving/serve.py (AnimaNode)
- [ ] T07.5: Update anima_module.yaml with docker/ros2 fields
- [ ] T07.6: Create HF push script
- [ ] T07.7: Verify docker build + /health endpoint
