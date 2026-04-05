# DarkDriving -- ANIMA Defense Module

## Paper
- **Title**: DarkDriving: A Real-World Day and Night Aligned Dataset for Autonomous Driving in the Dark Environment
- **ArXiv**: 2603.18067
- **Venue**: ICRA 2026
- **Type**: Dataset benchmark (not a novel model architecture)

## Summary
DarkDriving introduces a real-world benchmark dataset of 9,538 precisely aligned day-night
image pairs for evaluating low-light enhancement in autonomous driving. Data was collected
using an automatic Trajectory Tracking based Pose Matching (TTPM) method at a 69-acre
closed driving test field at Chang'an University. Alignment error is within centimeters. The
paper benchmarks existing low-light enhancement methods (Retinexformer, SNR-Aware,
LLFormer, etc.) and evaluates downstream 2D/3D detection (YOLOv11, BEVDepth, CRN).

## Architecture Overview
This is a **benchmark/dataset** paper. There is no single novel architecture. The module
implements:
1. **TTPM Pipeline**: NDT-based localization + Pure Pursuit trajectory tracking + Euclidean
   pose matching (delta <= 5cm)
2. **Enhancement Baselines**: Retinexformer, SNR-Aware, SNR-SKF, LLFormer, ControlNet,
   CLIP-LIT, EnlightenGAN, LightTheNight
3. **Detection Heads**: YOLOv11 (2D), BEVDepth (3D RGB-only), CRN (3D RGB+Radar)
4. **Evaluation**: full-reference (PSNR, SSIM, LPIPS) + no-reference (MUSIQ, NIQE,
   HyperIQA, CNNIQA) + detection (AP50, AP50-90, ATE, AOE)

## Dataset Specifications
- **Total pairs**: 9,538 day-night aligned (19,076 images)
- **Resolution**: 2448 x 2048 (raw), resized to 512 x 512 for enhancement training
- **Annotations**: 13,184 2D bounding boxes, single class (Car)
- **Splits**: train 5,906 pairs / test 3,632 pairs
- **Sensor suite**: RGB camera (41 deg FOV), 128-ch LiDAR (200m range, +/-2cm), Xsens MTi-680G (400Hz)
- **Road types (6)**: multi-lane, single-lane, curved, open, T-intersection, intersection
- **Lighting conditions (12)**: no streetlight, vehicle low/high beam, bilateral/unilateral
  streetlight combos, backlight

## Hyperparameters (Enhancement Training)
- **Input size**: 512 x 512
- **Optimizer**: Adam (beta1=0.9, beta2=0.999)
- **Learning rate**: 1e-3 with cosine annealing
- **Batch size**: 8 (paper default; auto-detect on L4)
- **Loss**: L1 (MAE) between enhanced and ground-truth day image
- **Augmentation**: random crop, rotation, horizontal flip
- **Precision**: mixed (AMP)
- **Epochs**: 200 (with early stopping patience 20)

## Detection Hyperparameters
- **YOLOv11**: COCO pretrained, finetuned on DarkDriving
- **BEVDepth**: nuScenes pretrained (RGB-only 3D)
- **CRN**: nuScenes pretrained (RGB+Radar 3D)

## Evaluation Metrics
| Category | Metric | Direction |
|----------|--------|-----------|
| Full-ref | PSNR | higher |
| Full-ref | SSIM | higher |
| Full-ref | LPIPS | lower |
| No-ref | MUSIQ | higher |
| No-ref | NIQE | lower |
| No-ref | HyperIQA | higher |
| No-ref | CNNIQA | higher |
| 2D det | AP50 | higher |
| 2D det | AP50-90 | higher |
| 3D det | AP | higher |
| 3D det | ATE | lower |
| 3D det | AOE | lower |

## Model Requirements
- Retinexformer checkpoint (from official repo or HF)
- YOLOv11 COCO pretrained (on disk: /mnt/forge-data/models/yolo11n.pt)
- BEVDepth nuScenes checkpoint
- CRN nuScenes checkpoint
- Optional: ControlNet (Stable Diffusion v1.5 based)

## Dataset Requirements
- DarkDriving dataset (9,538 pairs, ~20-40GB estimated)
- nuScenes (on disk: /mnt/forge-data/datasets/nuscenes/)
- COCO (on disk: /mnt/forge-data/datasets/coco/)

## Key Files
```
src/dark_driving/
  __init__.py          -- package init
  model.py             -- enhancement model wrappers (Retinexformer, SNR-Aware, etc.)
  dataset.py           -- DarkDriving dataset loader + augmentations
  train.py             -- training loop for enhancement models
  evaluate.py          -- full-ref + no-ref + detection evaluation
  losses.py            -- L1, SSIM, LPIPS, perceptual losses
  utils.py             -- metrics, visualization, config loading
  serve.py             -- AnimaNode for Docker serving
configs/
  paper.toml           -- paper-exact reproduction config
  debug.toml           -- quick smoke test (2 epochs)
  retinexformer.toml   -- Retinexformer-specific config
  detection.toml       -- downstream detection config
```
