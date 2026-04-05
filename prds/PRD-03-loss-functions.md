# PRD-03: Loss Functions

## Objective
Implement all loss functions used in the DarkDriving benchmark.

## Deliverables
1. `src/dark_driving/losses.py` containing:
   - **L1Loss** (MAE) -- primary enhancement loss
   - **SSIMLoss** -- structural similarity (differentiable, 1 - SSIM)
   - **LPIPSLoss** -- perceptual loss using VGG features
   - **PerceptualLoss** -- VGG-based feature matching loss
   - **CombinedLoss** -- configurable weighted sum of above losses
   - **CharbonnierLoss** -- smooth L1 variant (used by Retinexformer)

## Loss Equations
- L1: `L = mean(|enhanced - target|)`
- SSIM: `L = 1 - SSIM(enhanced, target)` (window_size=11)
- LPIPS: `L = LPIPS_VGG(enhanced, target)`
- Charbonnier: `L = mean(sqrt((enhanced - target)^2 + eps^2))`, eps=1e-3
- Combined: `L = w1*L1 + w2*SSIM + w3*LPIPS`

## Default Weights (paper config)
- L1 weight: 1.0
- SSIM weight: 0.0 (off by default, enable for ablations)
- LPIPS weight: 0.0 (off by default)

## Acceptance Criteria
- Each loss computes a scalar on random (B,3,H,W) inputs
- Gradients flow through all losses
- Combined loss respects config weights
