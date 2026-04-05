# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 90%

## Done
- [x] Paper analysis (arXiv 2603.18067 -- DarkDriving)
- [x] CLAUDE.md, ASSETS.md, PRD.md, 7 PRDs, task index
- [x] anima_module.yaml, pyproject.toml (torch cu128)
- [x] PRD-01 Foundation: venv, deps, configs
- [x] PRD-02 Core Model: Retinexformer, SNR-Aware, LLFormer, UNet
- [x] PRD-03 Loss Functions: L1, SSIM, LPIPS, Charbonnier, Combined
- [x] PRD-04 Training Pipeline: train.py + train_cuda.py
- [x] PRD-05 Evaluation: full-ref + no-ref + detection metrics
- [x] PRD-06 Export Pipeline: SafeTensors + ONNX + TRT FP16/FP32
- [x] PRD-07 Integration: Docker serve, docker-compose, .env.serve
- [x] CUDA kernels: fused ops copied to shared_infra
- [x] Fix: transposed attention (O(C^2) vs O((H*W)^2))
- [x] Fix: fp32 attention matmul (fp16 overflows on 262K sum)
- [x] Multi-source dataset: nuScenes day (26.8K) + KITTI (7.5K) = 34.3K pairs
- [x] Batch finder: bs=4 at fp32 fits 52% VRAM on L4
- [x] 63/63 tests pass, ruff clean
- [x] 7 focused git commits
- [x] [07:35] Training launched on GPU 1 (PID 25770)

## In Progress
- [ ] Full training: 100 epochs on multi-source data (GPU 1, ~150h ETA)
  - PID: 25770 | GPU: CUDA_VISIBLE_DEVICES=1
  - Log: /mnt/artifacts-datai/logs/project_darkdriving/train_20260405_0735.log
  - Config: configs/multi_source.toml (fp32, bs=4, lr=1e-3, cosine)
  - Data: 30,840 train + 3,426 val + 3,987 test (nuScenes night)
  - Speed: 1.36 it/s = ~1.5h/epoch, VRAM: 12.1GB/23GB (52.5%)

## TODO (after training completes)
- [ ] Run evaluation on test set (nuScenes night)
- [ ] Export: pth -> safetensors -> ONNX -> TRT FP16 + TRT FP32
- [ ] Docker build + health test
- [ ] Push to HuggingFace: ilessio-aiflowlab/project_darkdriving-checkpoint
- [ ] Retrain with DarkDriving dataset when released (paper's 9.5K pairs)

## Extra Data Strategy (surpass paper)
- nuScenes day (26.8K) + KITTI (7.5K) synthetic pairs = 34.3K total
- vs paper's 5,906 train pairs
- Real nuScenes night (3,987) for evaluation
- VIVID++ thermal (47GB) skipped — mostly thermal, sparse RGB

## Downloads Still Needed
- DarkDriving dataset -- ~30GB -- URL TBD
- Retinexformer pretrained -- github.com/caiyuanhao1998/Retinexformer
