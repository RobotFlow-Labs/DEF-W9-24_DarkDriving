# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 90%

## Done
- [x] All 7 PRDs built (63/63 tests, ruff clean)
- [x] Corrupt image fix (KITTI 007200.png — robust fallback)
- [x] Force pushed to origin main

## In Progress
- [ ] Training: PID 353964 | GPU 1 | 12.1GB/23GB | 1.38 it/s
  - Log: /mnt/artifacts-datai/logs/project_darkdriving/train_20260405_0927.log
  - Monitor: PID 358316 (auto-ship when done)
  - Config: fp32, bs=4, lr=1e-3, 100 epochs, early stop patience=15

## TODO (auto-ships when training completes)
- [ ] Export: pth -> safetensors -> ONNX -> TRT FP16 -> TRT FP32
- [ ] Git push + HF push (standing order — autonomous)
- [ ] Retrain with DarkDriving dataset when released
