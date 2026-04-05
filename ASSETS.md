# ASSETS.md -- DarkDriving Asset Inventory

## Datasets

### Required -- Must Download
| Asset | Size (est.) | Source | Local Path |
|-------|-------------|--------|------------|
| DarkDriving dataset | ~30GB | GitHub/project page (TBD) | /mnt/forge-data/datasets/darkdriving/ |

### Already On Disk -- DO NOT Download
| Asset | Size | Local Path |
|-------|------|------------|
| nuScenes | ~300GB | /mnt/forge-data/datasets/nuscenes/ |
| COCO val+train | ~40GB | /mnt/forge-data/datasets/coco/ |
| KITTI | ~50GB | /mnt/forge-data/datasets/kitti/ |
| nuScenes voxels cache | 163GB | /mnt/forge-data/shared_infra/datasets/nuscenes_voxels/ |
| nuScenes DINOv2 features | 140GB | /mnt/forge-data/shared_infra/datasets/nuscenes_dinov2_features/ |

## Pretrained Models

### Already On Disk -- DO NOT Download
| Model | Path |
|-------|------|
| YOLO11n | /mnt/forge-data/models/yolo11n.pt |
| YOLOv5l6 | /mnt/forge-data/models/yolov5l6.pt |
| Stable Diffusion 2.1 | /mnt/forge-data/models/stable-diffusion-2-1/ |
| DINOv2 ViT-B/14 | /mnt/forge-data/models/dinov2_vitb14_pretrain.pth |
| CLIP ViT-B/32 | /mnt/forge-data/models/clip-vit-base-patch32/ |

### Required -- Must Download
| Model | Size (est.) | Source | Local Path |
|-------|-------------|--------|------------|
| Retinexformer checkpoint | ~200MB | github.com/caiyuanhao1998/Retinexformer | /mnt/forge-data/models/retinexformer/ |
| BEVDepth nuScenes ckpt | ~500MB | official repo | /mnt/forge-data/models/bevdepth/ |
| CRN nuScenes ckpt | ~500MB | official repo | /mnt/forge-data/models/crn/ |
| SNR-Aware checkpoint | ~300MB | github.com/JIA-Lab-research/SNR-Aware-Low-Light-Enhance | /mnt/forge-data/models/snr_aware/ |

## Shared CUDA Kernels (from shared_infra)
| Kernel | Path | Used For |
|--------|------|----------|
| Fused image preprocess | cuda_extensions/fused_image_preprocess/ | Input normalization |
| Detection ops | cuda_extensions/detection_ops/ | YOLOv11 NMS |

## Output Paths
| Type | Path |
|------|------|
| Checkpoints | /mnt/artifacts-datai/checkpoints/project_darkdriving/ |
| Logs | /mnt/artifacts-datai/logs/project_darkdriving/ |
| TensorBoard | /mnt/artifacts-datai/tensorboard/project_darkdriving/ |
| Exports | /mnt/artifacts-datai/exports/project_darkdriving/ |
| Reports | /mnt/artifacts-datai/reports/project_darkdriving/ |

## Download Commands (run when approved)
```bash
# DarkDriving dataset -- URL TBD (paper says GitHub release)
# huggingface-cli download <repo> --local-dir /mnt/forge-data/datasets/darkdriving/

# Retinexformer
# git clone https://github.com/caiyuanhao1998/Retinexformer /mnt/forge-data/repos/retinexformer

# SNR-Aware
# git clone https://github.com/JIA-Lab-research/SNR-Aware-Low-Light-Enhance /mnt/forge-data/repos/snr_aware
```
