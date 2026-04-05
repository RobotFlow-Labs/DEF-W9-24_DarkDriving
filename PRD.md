# PRD.md -- DarkDriving Master Build Plan

## Module Identity
- **Name**: project_darkdriving
- **Paper**: DarkDriving (arXiv 2603.18067, ICRA 2026)
- **Type**: Dataset benchmark -- low-light enhancement for autonomous driving
- **Package**: dark_driving

## Build Plan

| PRD | Name | Status | Description |
|-----|------|--------|-------------|
| PRD-01 | Foundation | [x] DONE | Project structure, configs, dependencies, venv |
| PRD-02 | Core Model | [x] DONE | Enhancement model wrappers (Retinexformer, SNR-Aware, LLFormer, UNet) |
| PRD-03 | Loss Functions | [x] DONE | L1, SSIM, LPIPS, perceptual, Charbonnier, combined losses |
| PRD-04 | Training Pipeline | [x] DONE | Train loop, CUDA-accelerated, checkpointing, LR scheduling, AMP |
| PRD-05 | Evaluation | [x] DONE | Full-ref, no-ref, detection metrics pipeline |
| PRD-06 | Export Pipeline | [x] DONE | SafeTensors, ONNX, TensorRT FP16/FP32 export |
| PRD-07 | Integration | [x] DONE | Docker serve, ROS2 node, HF push ready |

## Architecture Summary
DarkDriving is a benchmark dataset paper. The module benchmarks existing low-light
enhancement methods on the DarkDriving dataset (9,538 day-night pairs, 2448x2048,
resized to 512x512) and evaluates downstream 2D/3D detection.

### Enhancement Pipeline
```
Night Image (512x512) -> Enhancement Model -> Enhanced Image -> Detection Model -> Predictions
                              |                     |
                         Retinexformer          PSNR/SSIM/LPIPS
                         SNR-Aware              MUSIQ/NIQE
                         LLFormer               AP50/AP50-90
                         ControlNet
```

### CUDA Acceleration
- torch.compile fused kernels for preprocessing, augmentation, metrics
- Fused L1+SSIM loss (single GPU kernel)
- Fused PSNR+SSIM metric computation
- GPU-native random flip/crop (zero CPU roundtrip)

### Key Numbers
- 9,538 train pairs, 3,632 test pairs
- 13,184 bounding box annotations (Car class)
- 6 road types, 12 lighting conditions
- Input: 512x512, LR: 1e-3, Batch: 8, Epochs: 200
