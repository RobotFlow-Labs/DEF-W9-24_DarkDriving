# PRD-05: Evaluation

## Objective
Implement the full evaluation pipeline matching the paper's metrics.

## Deliverables
1. `src/dark_driving/evaluate.py` -- Evaluation entry point:
   - Full-reference metrics: PSNR, SSIM, LPIPS
   - No-reference metrics: MUSIQ, NIQE, HyperIQA, CNNIQA
   - 2D detection: AP50, AP50-90 (via YOLOv11)
   - 3D detection: AP, ATE, AOE (via BEVDepth/CRN -- optional)
   - Per-lighting-condition breakdown (12 conditions)
   - Per-road-type breakdown (6 types)
   - JSON + markdown report generation
2. `scripts/evaluate.py` -- CLI wrapper

## Metrics Implementation
| Metric | Library | Notes |
|--------|---------|-------|
| PSNR | torchmetrics / skimage | Peak signal-to-noise ratio |
| SSIM | torchmetrics / skimage | Structural similarity |
| LPIPS | lpips package | VGG-based perceptual |
| MUSIQ | pyiqa | Multi-scale image quality |
| NIQE | pyiqa | Natural image quality |
| HyperIQA | pyiqa | Hyper-network IQA |
| CNNIQA | pyiqa | CNN-based IQA |
| AP50 | torchmetrics.detection | COCO-style AP@0.5 |

## Evaluation Flow
```
1. Load enhanced images + ground truth day images
2. Compute full-reference metrics (PSNR, SSIM, LPIPS)
3. Compute no-reference metrics on enhanced images
4. Run YOLOv11 on enhanced images -> compute AP
5. (Optional) Run BEVDepth/CRN for 3D metrics
6. Aggregate by lighting condition + road type
7. Generate report
```

## Acceptance Criteria
- Each metric computes on a pair of random images
- Report is saved to /mnt/artifacts-datai/reports/project_darkdriving/
- Metrics match expected ranges on known inputs
