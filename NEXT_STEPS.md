# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 85%

## Done
- [x] Paper analysis (arXiv 2603.18067 -- DarkDriving)
- [x] CLAUDE.md -- paper summary, architecture, hyperparameters
- [x] ASSETS.md -- asset inventory with paths
- [x] PRD.md -- master build plan with 7 PRDs
- [x] prds/ -- 7 PRD files (PRD-01 through PRD-07)
- [x] tasks/INDEX.md -- granular task list
- [x] anima_module.yaml -- module manifest
- [x] pyproject.toml -- hatchling backend, torch cu128, onnxscript
- [x] configs/paper.toml -- paper-exact hyperparameters
- [x] configs/debug.toml -- smoke test config (2 epochs)
- [x] configs/retinexformer.toml -- Retinexformer-specific config
- [x] configs/detection.toml -- downstream detection config
- [x] PRD-01 Foundation: venv created, deps installed, import verified
- [x] PRD-02 Core Model: Retinexformer, SNR-Aware, LLFormer, UNet (22 tests pass)
- [x] PRD-03 Loss Functions: L1, SSIM, LPIPS, Charbonnier, Combined (14 tests)
- [x] PRD-04 Training Pipeline: train.py + train_cuda.py, checkpointing, LR scheduling
- [x] PRD-05 Evaluation: full-ref + no-ref + detection metrics
- [x] PRD-06 Export Pipeline: SafeTensors + ONNX + TRT FP16/FP32
- [x] PRD-07 Integration: Docker serve, docker-compose, .env.serve
- [x] CUDA kernels: fused_preprocess, fused_flip, fused_crop, fused_psnr_ssim, fused_loss
- [x] Batch size finder: scripts/find_batch_size.py
- [x] 63/63 tests pass, ruff clean
- [x] Dockerfile.serve + docker-compose.serve.yml

## In Progress
- [ ] Nothing currently in progress

## TODO
- [ ] Download DarkDriving dataset (URL TBD from paper release)
- [ ] Download Retinexformer pretrained checkpoint
- [ ] Download BEVDepth + CRN checkpoints for 3D evaluation
- [ ] Run smoke test on GPU (configs/debug.toml)
- [ ] Run full training (configs/paper.toml) -- needs GPU + dataset
- [ ] Run evaluation pipeline on test set
- [ ] Export to ONNX + TensorRT (post-training)
- [ ] Copy CUDA kernels to shared_infra/cuda_extensions/
- [ ] Docker build + health test
- [ ] Push to HuggingFace: ilessio-aiflowlab/project_darkdriving-checkpoint
- [ ] Extended training with nuScenes night + VIVID++ thermal + KITTI

## Blocking
- DarkDriving dataset not yet available (paper says GitHub release TBD)
- Training blocked on GPU allocation (ask user before starting)

## Extra Data Strategy (surpass paper)
- nuScenes (479GB) at /mnt/forge-data/datasets/nuscenes/ — day+night scenes
- KITTI (52GB) at /mnt/forge-data/datasets/kitti/ — daytime baseline
- VIVID++ thermal (47GB) at /mnt/train-data/datasets/vivid_plus_plus/ — night thermal
- Pre-computed: nuScenes DINOv2 + KITTI DINOv2 in shared_infra

## Downloads Needed
- DarkDriving dataset -- ~30GB -- URL TBD (paper GitHub release)
- Retinexformer checkpoint -- ~200MB -- github.com/caiyuanhao1998/Retinexformer
- BEVDepth nuScenes ckpt -- ~500MB -- official repo
- CRN nuScenes ckpt -- ~500MB -- official repo
- SNR-Aware checkpoint -- ~300MB -- github.com/JIA-Lab-research/SNR-Aware-Low-Light-Enhance
