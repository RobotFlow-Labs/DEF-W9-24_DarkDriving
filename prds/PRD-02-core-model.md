# PRD-02: Core Model

## Objective
Implement enhancement model wrappers and the DarkDriving dataset loader.

## Deliverables
1. `src/dark_driving/model.py` -- Enhancement model registry with:
   - Retinexformer wrapper (ORF architecture: illumination estimator + IGT restoration)
   - SNR-Aware wrapper (SNR-guided transformer + CNN hybrid)
   - LLFormer wrapper (axis-based multi-head attention)
   - Generic UNet baseline for ablations
2. `src/dark_driving/dataset.py` -- DarkDriving PyTorch Dataset:
   - Load day-night aligned pairs from directory structure
   - Resize to 512x512
   - Random crop, rotation, horizontal flip augmentations
   - Train/test split per paper (5,906 / 3,632)
   - 2D bbox annotation loading (YOLO or COCO format)
3. `tests/test_model.py` -- Smoke tests for model forward pass
4. `tests/test_dataset.py` -- Dataset loading tests

## Architecture: Retinexformer (Primary Baseline)
```
Input (B, 3, 512, 512)
  -> Illumination Estimator (shallow CNN, outputs illumination map)
  -> Element-wise: lit_image = input * illumination_map
  -> Illumination-Guided Transformer (IGT):
       Multiple IG-MSA blocks (illumination-guided multi-head self-attention)
       Feed-forward with GELU
       Skip connections
  -> Output (B, 3, 512, 512) -- enhanced image
```

## Acceptance Criteria
- `model.forward(torch.randn(1, 3, 512, 512))` returns tensor of same shape
- Dataset loads at least 1 sample without error (or gracefully skips if data absent)
- All tests pass with `uv run pytest tests/ -x -v`
